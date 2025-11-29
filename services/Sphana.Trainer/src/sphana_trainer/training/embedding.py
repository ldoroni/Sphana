"""Embedding training workflow."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from sphana_trainer.config import EmbeddingConfig
from sphana_trainer.data import EmbeddingPairDataset, build_embedding_collate_fn
from sphana_trainer.exporters.onnx import export_embedding_encoder
from sphana_trainer.models import EmbeddingEncoder
from sphana_trainer.training.context import create_trainer_context, save_training_state, load_training_state
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
from sphana_trainer.training.profiling import ProfilerManager
from sphana_trainer.utils.progress import ProgressTracker, _format_duration


@dataclass
class EmbeddingTrainingResult:
    checkpoint_dir: Path
    onnx_path: Path
    quantized_path: Path
    metrics: Dict[str, float]


class EmbeddingTrainer:
    def __init__(self, config: EmbeddingConfig, artifact_root: Path) -> None:
        self.config = config
        self.artifact_root = artifact_root
        self.ctx = create_trainer_context(config.ddp, config.precision)
        seed_everything(config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        base_model = EmbeddingEncoder.from_pretrained(config.model_name, pooling=config.pooling_strategy)
        self.model = base_model.to(self.ctx.device)
        self.model = self.ctx.wrap_model(self.model)
        self.telemetry = TelemetryMonitor(self.ctx.device)

    def train(self) -> EmbeddingTrainingResult:
        train_file = resolve_split_file(self.config.dataset_path, self.config.train_file, "train")
        if not train_file:
            raise FileNotFoundError(f"Unable to resolve training dataset from {self.config.dataset_path}")
        val_file = (
            resolve_split_file(
                self.config.validation_path or self.config.dataset_path,
                self.config.validation_file,
                "validation",
            )
            if self.config.validation_path or self.config.validation_file or self.config.dataset_path.is_dir()
            else None
        )

        train_dataset = EmbeddingPairDataset(train_file)
        val_dataset = EmbeddingPairDataset(val_file, limit=self.config.eval_subset_size) if val_file else None

        if self.config.metric_threshold is not None and val_dataset is None:
            raise RuntimeError("Embedding metric_threshold requires a validation split.")

        component_dir = Path(self.config.output_dir)
        run_dir = prepare_run_directory(component_dir, self.config.version)
        metrics_logger = MetricsLogger(run_dir, "embedding", run_dir.name) if self.ctx.is_primary else None
        mlflow_context = nullcontext()
        if self.ctx.is_primary and self.config.log_to_mlflow:
            mlflow_context = MlflowLogger(
                True,
                tracking_uri=self.config.mlflow_tracking_uri,
                experiment=self.config.mlflow_experiment,
                run_name=self.config.mlflow_run_name or f"embedding-{run_dir.name}",
                tags={"component": "embedding", "run_id": run_dir.name},
            )
        dataset_hash = dataset_fingerprint(Path(self.config.dataset_path))
        if self.ctx.is_primary:
            device_name = (
                torch.cuda.get_device_name(self.ctx.device) if self.ctx.device.type == "cuda" else str(self.ctx.device)
            )
            logger.info(
                "Embedding run_id={} dataset_hash={} device={} precision={} world_size={}",
                run_dir.name,
                dataset_hash,
                device_name,
                self.ctx.precision,
                self.ctx.world_size,
            )

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
            batch_size=self.config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=build_embedding_collate_fn(self.tokenizer, self.config.max_seq_length),
        )
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=build_embedding_collate_fn(self.tokenizer, self.config.max_seq_length),
            )

        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        total_steps = len(train_loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )
        self._maybe_resume(optimizer, scheduler)

        best_val = float("-inf")
        total_examples = len(train_dataset)
        metrics: Dict[str, float] = {}

        profiler = ProfilerManager(self.ctx.device, self.config.profile_steps, run_dir / "profile.json")

        with mlflow_context as mlflow_run:
            if mlflow_run:
                mlflow_run.log_params(
                    {
                        "model_name": self.config.model_name,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                        "epochs": self.config.epochs,
                        "max_seq_length": self.config.max_seq_length,
                        "precision": self.ctx.precision,
                        "gradient_accumulation": self.config.gradient_accumulation,
                        "dataset_hash": dataset_hash,
                    }
                )

            logger.info("Starting training: {} epochs, {} batches per epoch", self.config.epochs, len(train_loader))
            
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
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    with self.ctx.autocast():
                        loss = self._contrastive_step(batch)
                    if self.ctx.scaler:
                        self.ctx.scaler.scale(loss).backward()
                        self.ctx.scaler.step(optimizer)
                        self.ctx.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    running_loss += loss.item()
                    batch_progress.update()
                    
                avg_loss = running_loss / max(1, len(train_loader))
                if val_loader:
                    val_metric = self._evaluate(val_loader)
                    best_val = max(best_val, val_metric)
                else:
                    val_metric = 0.0
                
                # Enhanced epoch complete logging
                duration = max(perf_counter() - epoch_start, 1e-9)
                examples_per_sec = (total_examples / duration) if total_examples else 0.0
                logger.info(
                    "Epoch {}/{} complete | Train loss: {:.4f} | Val metric: {:.4f} | "
                    "Epoch time: {} | Throughput: {:.2f} examples/sec",
                    epoch + 1,
                    self.config.epochs,
                    avg_loss,
                    val_metric,
                    _format_duration(duration),
                    examples_per_sec
                )
                
                if metrics_logger:
                    metadata = {"stage": "train", **self._system_metadata()}
                    metrics_logger.log(
                        epoch,
                        {
                            "train_loss": float(avg_loss),
                            "epoch_duration": duration,
                            "examples_per_sec": examples_per_sec,
                        },
                        metadata,
                    )
                    if val_loader:
                        metrics_logger.log(
                            epoch,
                            {"val_cosine": float(val_metric)},
                            {"stage": "validation", **self._system_metadata()},
                        )
                if mlflow_run:
                    mlflow_run.log_metrics(
                        {"train_loss": float(avg_loss), **self._telemetry_metrics()},
                        step=epoch,
                    )
                    if val_loader:
                        mlflow_run.log_metrics(
                            {"val_cosine": float(val_metric), **self._telemetry_metrics()},
                            step=epoch,
                        )
                if profiler.enabled:
                    profiler.step()

            metrics = {"train_loss": float(avg_loss)}
            if val_loader:
                metrics["val_cosine"] = float(val_metric)
                metrics["best_val_cosine"] = float(best_val)
                self._ensure_metric_threshold(best_val)
            if mlflow_run:
                mlflow_run.log_metrics({**metrics, **self._telemetry_metrics()}, step=self.config.epochs)
        profiler.stop()

        if self.ctx.ddp:  # pragma: no cover - requires distributed setup
            self.ctx.barrier()
            if not self.ctx.is_primary:
                self.ctx.cleanup()
                return EmbeddingTrainingResult(
                    checkpoint_dir=Path(self.config.output_dir),
                    onnx_path=Path(),
                    quantized_path=Path(),
                    metrics=metrics,
                )

        checkpoint_dir = run_dir / "checkpoint"
        model_dir = checkpoint_dir / "model"
        tokenizer_dir = checkpoint_dir / "tokenizer"
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self._unwrap_model()
        model_to_save.base_model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)
        (checkpoint_dir / "pooling.json").write_text(
            json.dumps({"pooling": self.config.pooling_strategy}, indent=2)
        )

        onnx_dir = run_dir / "onnx"
        onnx_path, quantized_path = export_embedding_encoder(
            model_to_save,
            self.tokenizer,
            onnx_dir,
            self.config.export_opset,
            self.config.max_seq_length,
            self.config.quantize,
        )
        save_training_state(checkpoint_dir / "state.pt", self.model, optimizer, scheduler, self.ctx.scaler)

        metadata = build_metadata(
            component="embedding",
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
        return EmbeddingTrainingResult(
            checkpoint_dir=checkpoint_dir,
            onnx_path=onnx_path,
            quantized_path=quantized_path,
            metrics=metrics,
        )

    def _contrastive_step(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        anchor = {k: v.to(self.ctx.device) for k, v in batch["anchor"].items()}
        positive = {k: v.to(self.ctx.device) for k, v in batch["positive"].items()}
        anchor_emb = self.model(**anchor)
        positive_emb = self.model(**positive)
        return multiple_negatives_loss(anchor_emb, positive_emb, self.config.temperature)

    def _evaluate(self, loader: DataLoader) -> float:
        logger.info("Running validation on {} batches", len(loader))
        self.model.eval()
        sims = []
        
        val_progress = ProgressTracker(
            total=len(loader),
            stage_name="Validation",
            total_stages=1,
            current_stage=1,
            log_interval=self.config.progress_log_interval,
        )
        
        with torch.no_grad():
            for batch in loader:
                anchor = {k: v.to(self.ctx.device) for k, v in batch["anchor"].items()}
                positive = {k: v.to(self.ctx.device) for k, v in batch["positive"].items()}
                anchor_emb = self.model(**anchor)
                positive_emb = self.model(**positive)
                sim = F.cosine_similarity(anchor_emb, positive_emb).mean().item()
                sims.append(sim)
                val_progress.update()
        
        avg_sim = sum(sims) / max(1, len(sims))
        logger.info("Validation complete | Cosine similarity: {:.4f}", avg_sim)
        return avg_sim

    def _maybe_resume(self, optimizer, scheduler) -> None:
        if not self.config.resume_from:
            return
        state_file = Path(self.config.resume_from) / "checkpoint" / "state.pt"
        if not state_file.exists():
            return
        load_training_state(state_file, self.model, optimizer, scheduler, self.ctx.scaler)

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _ensure_metric_threshold(self, metric_value: float) -> None:
        if self.config.metric_threshold is None:
            return
        if metric_value < self.config.metric_threshold:
            raise RuntimeError(
                f"Embedding validation metric {metric_value:.4f} did not meet threshold "
                f"{self.config.metric_threshold:.4f}"
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


def multiple_negatives_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = torch.mm(anchor, positive.t()) / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return (loss_a + loss_b) * 0.5


