"""GNN ranking training workflow."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sphana_trainer.config import GNNConfig
from sphana_trainer.data import GraphQueryDataset
from sphana_trainer.exporters.onnx import export_gnn_ranker
from sphana_trainer.models import GGNNRanker
from sphana_trainer.training.context import create_trainer_context, save_training_state, load_training_state
from sphana_trainer.training.profiling import ProfilerManager
from sphana_trainer.utils import (
    build_metadata,
    prepare_run_directory,
    prune_run_directories,
    resolve_split_file,
    save_artifact_metadata,
    seed_everything,
)
from sphana_trainer.utils.fingerprint import dataset_fingerprint
from sphana_trainer.utils.metrics import MetricsLogger
from sphana_trainer.utils.mlflow import MlflowLogger
from sphana_trainer.utils.telemetry import TelemetryMonitor
from sphana_trainer.utils.telemetry import TelemetryMonitor
from sphana_trainer.utils.progress import ProgressTracker, _format_duration


@dataclass
class GNNTrainingResult:
    checkpoint_dir: Path
    onnx_path: Path
    quantized_path: Path
    metrics: Dict[str, float]


class GNNTrainer:
    def __init__(self, config: GNNConfig, artifact_root: Path) -> None:
        self.config = config
        self.artifact_root = artifact_root
        self.ctx = create_trainer_context(config.ddp, config.precision)
        seed_everything(config.seed)
        self.telemetry = TelemetryMonitor(self.ctx.device)
        self.telemetry = TelemetryMonitor(self.ctx.device)

    def train(self) -> GNNTrainingResult:
        train_file = resolve_split_file(self.config.dataset_path, self.config.train_file, "train")
        if not train_file:
            raise FileNotFoundError(f"GNN training file not found under {self.config.dataset_path}")
        val_file = resolve_split_file(
            self.config.validation_path or self.config.dataset_path,
            self.config.validation_file,
            "validation",
        )
        train_dataset = GraphQueryDataset(train_file)
        val_dataset = GraphQueryDataset(val_file) if val_file else None

        if self.config.metric_threshold is not None and val_dataset is None:
            raise RuntimeError("GNN metric_threshold requires a validation split.")

        component_dir = Path(self.config.output_dir)
        run_dir = prepare_run_directory(component_dir, self.config.version)
        metrics_logger = MetricsLogger(run_dir, "gnn", run_dir.name) if self.ctx.is_primary else None
        mlflow_context = nullcontext()
        if self.ctx.is_primary and self.config.log_to_mlflow:
            mlflow_context = MlflowLogger(
                True,
                tracking_uri=self.config.mlflow_tracking_uri,
                experiment=self.config.mlflow_experiment,
                run_name=self.config.mlflow_run_name or f"gnn-{run_dir.name}",
                tags={"component": "gnn", "run_id": run_dir.name},
            )
        dataset_hash = dataset_fingerprint(Path(self.config.dataset_path))
        if self.ctx.is_primary:
            device_name = (
                torch.cuda.get_device_name(self.ctx.device) if self.ctx.device.type == "cuda" else str(self.ctx.device)
            )
            logger.info(
                "GNN run_id={} dataset_hash={} device={} precision={} world_size={}",
                run_dir.name,
                dataset_hash,
                device_name,
                self.ctx.precision,
                self.ctx.world_size,
            )
        profiler = ProfilerManager(self.ctx.device, self.config.profile_steps, run_dir / "profile.json")

        input_dim = len(train_dataset[0]["candidates"][0]["node_features"][0])
        model = GGNNRanker(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.ctx.device)
        model = self.ctx.wrap_model(model)
        self.model = model

        train_sampler = None
        if self.ctx.ddp:  # pragma: no cover - requires distributed setup
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.ctx.world_size,
                rank=self.ctx.global_rank,
                shuffle=True,
                drop_last=False,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=lambda batch: batch[0],
        )
        val_loader = (
            DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])
            if val_dataset
            else None
        )

        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        best_val = float("inf")
        self._maybe_resume(self.model, optimizer)

        total_examples = len(train_dataset)
        metrics: Dict[str, float] = {}

        with mlflow_context as mlflow_run:
            if mlflow_run:
                mlflow_run.log_params(
                    {
                        "model_name": self.config.model_name,
                        "learning_rate": self.config.learning_rate,
                        "epochs": self.config.epochs,
                        "hidden_dim": self.config.hidden_dim,
                        "num_layers": self.config.num_layers,
                        "max_nodes": self.config.max_nodes,
                        "max_edges": self.config.max_edges,
                        "temperature": self.config.temperature,
                        "dataset_hash": dataset_hash,
                    }
                )

            logger.info("Starting training: {} epochs, {} queries per epoch", self.config.epochs, len(train_loader))
            
            for epoch in range(self.config.epochs):
                epoch_start = perf_counter()
                logger.info("Epoch {}/{} | Starting training phase", epoch + 1, self.config.epochs)
                self.model.train()
                if train_sampler:  # pragma: no cover - requires distributed sampler
                    train_sampler.set_epoch(epoch)
                running_loss = 0.0
                
                # Add batch progress tracker
                batch_progress = ProgressTracker(
                    total=len(train_loader),
                    stage_name=f"Training epoch {epoch+1}",
                    total_stages=self.config.epochs,
                    current_stage=epoch+1,
                    log_interval=self.config.progress_log_interval,
                )
                
                for query in train_loader:
                    optimizer.zero_grad()
                    scores, labels = self._forward_candidates(self.model, query["candidates"])
                    with self.ctx.autocast():
                        loss = listnet_loss(scores, labels, self.config.temperature)
                    if self.ctx.scaler:
                        self.ctx.scaler.scale(loss).backward()
                        self.ctx.scaler.step(optimizer)
                        self.ctx.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    running_loss += loss.item()
                    batch_progress.update()
                    
                avg_loss = running_loss / max(1, len(train_loader))

                val_loss = None
                if val_loader:
                    val_loss = self._evaluate(self.model, val_loader)
                    if val_loss < best_val:
                        best_val = val_loss
                
                # Enhanced epoch complete logging
                duration = max(perf_counter() - epoch_start, 1e-9)
                examples_per_sec = (total_examples / duration) if total_examples else 0.0
                logger.info(
                    "Epoch {}/{} complete | Train loss: {:.4f} | Val loss: {:.4f} | "
                    "Epoch time: {} | Throughput: {:.2f} examples/sec",
                    epoch + 1,
                    self.config.epochs,
                    avg_loss,
                    val_loss if val_loss is not None else 0.0,
                    _format_duration(duration),
                    examples_per_sec
                )
                
                if metrics_logger:
                    metrics_logger.log(
                        epoch,
                        {
                            "train_loss": float(avg_loss),
                            "epoch_duration": duration,
                            "examples_per_sec": examples_per_sec,
                        },
                        {"stage": "train", **self._system_metadata()},
                    )
                    if val_loss is not None:
                        metrics_logger.log(
                            epoch, {"val_loss": float(val_loss)}, {"stage": "validation", **self._system_metadata()}
                        )
                if mlflow_run:
                    log_payload = {"train_loss": float(avg_loss)}
                    if val_loss is not None:
                        log_payload["val_loss"] = float(val_loss)
                    log_payload.update(self._telemetry_metrics())
                    mlflow_run.log_metrics(log_payload, step=epoch)
                if profiler.enabled:
                    profiler.step()

            metrics = {"train_loss": float(avg_loss)}
            if val_loader:
                metrics["val_loss"] = float(best_val)
                self._ensure_metric_threshold(best_val)
            if mlflow_run:
                mlflow_run.log_metrics({**metrics, **self._telemetry_metrics()}, step=self.config.epochs)
        profiler.stop()


        if self.ctx.ddp:  # pragma: no cover - requires distributed setup
            self.ctx.barrier()
            if not self.ctx.is_primary:
                self.ctx.cleanup()
                return GNNTrainingResult(
                    checkpoint_dir=Path(self.config.output_dir),
                    onnx_path=Path(),
                    quantized_path=Path(),
                    metrics=metrics,
                )

        checkpoint_dir = run_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = self._unwrap_model()
        model_path = model_to_save.save(checkpoint_dir)
        model_to_save.save_config(checkpoint_dir)

        onnx_dir = run_dir / "onnx"
        onnx_path, quantized_path = export_gnn_ranker(
            model_to_save,
            onnx_dir,
            self.config.export_opset,
            self.config.max_nodes,
            self.config.max_edges,
            self.config.quantize,
        )
        save_training_state(checkpoint_dir / "state.pt", self.model, optimizer, None, self.ctx.scaler)

        metadata = build_metadata(
            component="gnn",
            version=run_dir.name,
            checkpoint_dir=checkpoint_dir,
            onnx_path=onnx_path,
            quantized_path=quantized_path,
            metrics=metrics,
            config=self.config.model_dump(mode="json"),
        )
        save_artifact_metadata(metadata, Path(self.artifact_root))
        prune_run_directories(component_dir, self.config.max_checkpoints)

        self.ctx.cleanup()
        return GNNTrainingResult(
            checkpoint_dir=checkpoint_dir,
            onnx_path=onnx_path,
            quantized_path=quantized_path,
            metrics=metrics,
        )

    def _forward_candidates(
        self,
        model: GGNNRanker,
        candidates: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = []
        labels = []
        for candidate in candidates:
            node_features = torch.tensor(candidate["node_features"], dtype=torch.float32, device=self.ctx.device)
            edge_index = torch.tensor(candidate.get("edge_index", []), dtype=torch.long, device=self.ctx.device)
            if edge_index.numel() == 0:
                edge_index = torch.zeros((0, 2), dtype=torch.long, device=self.ctx.device)
            else:
                if edge_index.dim() == 1:
                    new_size = edge_index.numel() // 2
                    edge_index = edge_index[: new_size * 2].view(-1, 2)
                elif edge_index.dim() == 2 and edge_index.size(-1) != 2:
                    if edge_index.size(0) == 2:
                        edge_index = edge_index.t().contiguous()
                    else:
                        edge_index = edge_index.reshape(-1, 2)
            edge_dirs = torch.tensor(candidate.get("edge_directions", []), dtype=torch.long, device=self.ctx.device)
            if edge_dirs.numel() == 0 and edge_index.numel() > 0:
                edge_dirs = torch.zeros(edge_index.size(0), dtype=torch.long, device=self.ctx.device)
            score = model(node_features, edge_index, edge_dirs)
            scores.append(score)
            labels.append(candidate.get("label", 0.0))
        return torch.stack(scores), torch.tensor(labels, dtype=torch.float32, device=self.ctx.device)

    def _evaluate(self, model: GGNNRanker, loader: DataLoader) -> float:
        logger.info("Running validation on {} batches", len(loader))
        model.eval()
        losses = []
        
        val_progress = ProgressTracker(
            total=len(loader),
            stage_name="Validation",
            total_stages=1,
            current_stage=1,
            log_interval=self.config.progress_log_interval,
        )
        
        with torch.no_grad():
            for query in loader:
                scores, labels = self._forward_candidates(model, query["candidates"])
                losses.append(listnet_loss(scores, labels, self.config.temperature).item())
                val_progress.update()
        
        avg_loss = sum(losses) / max(1, len(losses))
        logger.info("Validation complete | Loss: {:.4f}", avg_loss)
        return avg_loss

    def _maybe_resume(self, model, optimizer) -> None:
        state_path = None
        if self.config.resume_from:
            state_path = Path(self.config.resume_from) / "state.pt"
        if state_path and state_path.exists():
            load_training_state(state_path, model, optimizer, None, self.ctx.scaler)

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _ensure_metric_threshold(self, metric_value: float) -> None:
        if self.config.metric_threshold is None:
            return
        if metric_value > self.config.metric_threshold:
            raise RuntimeError(
                f"GNN validation loss {metric_value:.4f} exceeded threshold {self.config.metric_threshold:.4f}"
            )

    def _system_metadata(self) -> Dict[str, float]:
        return self.telemetry.snapshot()

    def _telemetry_metrics(self) -> Dict[str, float]:
        snapshot = self.telemetry.snapshot()
        metrics = {}
        for key, value in snapshot.items():
            if isinstance(value, (int, float)):
                metrics[f"telemetry_{key}"] = float(value)
        return metrics


def listnet_loss(scores: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    pred = F.log_softmax(scores / temperature, dim=0)
    target = F.softmax(labels / temperature, dim=0)
    return -(target * pred).sum()


