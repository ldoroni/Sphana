"""Relation extraction training workflow."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from sphana_trainer.config import RelationExtractionConfig
from sphana_trainer.data import RelationDataset
from sphana_trainer.exporters.onnx import export_relation_classifier
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
class RelationTrainingResult:
    checkpoint_dir: Path
    onnx_path: Path
    quantized_path: Path
    metrics: Dict[str, float]


class RelationExtractionTrainer:
    def __init__(self, config: RelationExtractionConfig, artifact_root: Path) -> None:
        self.config = config
        self.artifact_root = artifact_root
        self.ctx = create_trainer_context(config.ddp, config.precision)
        seed_everything(config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        self.telemetry = TelemetryMonitor(self.ctx.device)
        self.telemetry = TelemetryMonitor(self.ctx.device)

    def train(self) -> RelationTrainingResult:
        train_file = resolve_split_file(self.config.dataset_path, self.config.train_file, "train")
        if not train_file:
            raise FileNotFoundError(f"Unable to resolve training dataset for {self.config.dataset_path}")

        val_file = resolve_split_file(
            self.config.validation_path or self.config.dataset_path,
            self.config.validation_file,
            "validation",
        )

        label2id: Dict[str, int] = {}
        train_dataset = RelationDataset(
            train_file,
            self.tokenizer,
            label2id=label2id,
            max_length=self.config.max_seq_length,
            allow_new_labels=True,
        )
        val_dataset = (
            RelationDataset(
                val_file,
                self.tokenizer,
                label2id=label2id,
                max_length=self.config.max_seq_length,
                allow_new_labels=True,
            )
            if val_file
            else None
        )

        if self.config.metric_threshold is not None and val_dataset is None:
            raise RuntimeError("Relation metric_threshold requires a validation split.")

        component_dir = Path(self.config.output_dir)
        run_dir = prepare_run_directory(component_dir, self.config.version)
        metrics_logger = MetricsLogger(run_dir, "relation", run_dir.name) if self.ctx.is_primary else None
        mlflow_context = nullcontext()
        if self.ctx.is_primary and self.config.log_to_mlflow:
            mlflow_context = MlflowLogger(
                True,
                tracking_uri=self.config.mlflow_tracking_uri,
                experiment=self.config.mlflow_experiment,
                run_name=self.config.mlflow_run_name or f"relation-{run_dir.name}",
                tags={"component": "relation", "run_id": run_dir.name},
            )
        dataset_hash = dataset_fingerprint(Path(self.config.dataset_path))
        if self.ctx.is_primary:
            device_name = (
                torch.cuda.get_device_name(self.ctx.device) if self.ctx.device.type == "cuda" else str(self.ctx.device)
            )
            logger.info(
                "Relation run_id={} dataset_hash={} device={} precision={} world_size={}",
                run_dir.name,
                dataset_hash,
                device_name,
                self.ctx.precision,
                self.ctx.world_size,
            )

        profiler = ProfilerManager(self.ctx.device, self.config.profile_steps, run_dir / "profile.json")

        id2label = {idx: label for label, idx in label2id.items()}
        with _suppress_hf_mismatch_warning():
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                problem_type="single_label_classification",
                ignore_mismatched_sizes=True,
            ).to(self.ctx.device)
        model = self.ctx.wrap_model(model)

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
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
            )
            if val_dataset
            else None
        )

        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        total_steps = len(train_loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )
        self._maybe_resume(model, optimizer, scheduler)

        best_f1 = 0.0
        epochs_without_improve = 0
        final_metrics: Dict[str, float] = {}
        total_examples = len(train_dataset)

        with mlflow_context as mlflow_run:
            if mlflow_run:
                mlflow_run.log_params(
                    {
                        "model_name": self.config.model_name,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                        "epochs": self.config.epochs,
                        "max_seq_length": self.config.max_seq_length,
                        "eval_batch_size": self.config.eval_batch_size,
                        "weight_decay": self.config.weight_decay,
                        "gradient_accumulation": self.config.gradient_accumulation,
                        "dataset_hash": dataset_hash,
                    }
                )

            logger.info("Starting training: {} epochs, {} batches per epoch", self.config.epochs, len(train_loader))
            
            for epoch in range(self.config.epochs):
                epoch_start = perf_counter()
                logger.info("Epoch {}/{} | Starting training phase", epoch + 1, self.config.epochs)
                model.train()
                if train_sampler:  # pragma: no cover - requires distributed sampler
                    train_sampler.set_epoch(epoch)
                running_loss = 0.0
                optimizer.zero_grad()
                
                # Add batch progress tracker
                batch_progress = ProgressTracker(
                    total=len(train_loader),
                    stage_name=f"Training epoch {epoch+1}",
                    total_stages=self.config.epochs,
                    current_stage=epoch+1,
                    log_interval=self.config.progress_log_interval,
                )
                
                for step, batch in enumerate(train_loader, start=1):
                    batch = {k: v.to(self.ctx.device) for k, v in batch.items()}
                    with self.ctx.autocast():
                        outputs = model(**batch)
                        loss = outputs.loss / self.config.gradient_accumulation
                    if self.ctx.scaler:
                        self.ctx.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if step % self.config.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if self.ctx.scaler:
                            self.ctx.scaler.step(optimizer)
                            self.ctx.scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    running_loss += loss.item()
                    batch_progress.update()
                    
                metrics = {"train_loss": running_loss / max(1, len(train_loader))}
                if val_loader:
                    eval_metrics = self._evaluate(model, val_loader)
                    metrics.update(eval_metrics)
                    if eval_metrics["macro_f1"] > best_f1:
                        best_f1 = eval_metrics["macro_f1"]
                        epochs_without_improve = 0
                    else:
                        epochs_without_improve += 1
                
                # Enhanced epoch complete logging
                duration = max(perf_counter() - epoch_start, 1e-9)
                examples_per_sec = (total_examples / duration) if total_examples else 0.0
                logger.info(
                    "Epoch {}/{} complete | Train loss: {:.4f} | F1: {:.4f} | "
                    "Epoch time: {} | Throughput: {:.2f} examples/sec",
                    epoch + 1,
                    self.config.epochs,
                    metrics["train_loss"],
                    metrics.get("macro_f1", 0.0),
                    _format_duration(duration),
                    examples_per_sec
                )
                
                if metrics_logger:
                    metrics_logger.log(
                        epoch,
                        {
                            "train_loss": metrics["train_loss"],
                            "epoch_duration": duration,
                            "examples_per_sec": examples_per_sec,
                        },
                        {"stage": "train", **self._system_metadata()},
                    )
                    if "macro_f1" in metrics:
                        metrics_logger.log(
                            epoch, {"macro_f1": metrics["macro_f1"]}, {"stage": "validation", **self._system_metadata()}
                        )
                if mlflow_run:
                    mlflow_run.log_metrics({**metrics, **self._telemetry_metrics()}, step=epoch)
                final_metrics = metrics
                if profiler.enabled:
                    profiler.step()
                if val_loader and epochs_without_improve >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping relation after %s epochs without improvement.", epochs_without_improve
                    )
                    break
            if mlflow_run and final_metrics:
                mlflow_run.log_metrics({**final_metrics, **self._telemetry_metrics()}, step=self.config.epochs)

        profiler.stop()

        if self.ctx.ddp:  # pragma: no cover - requires distributed setup
            self.ctx.barrier()
            if not self.ctx.is_primary:
                self.ctx.cleanup()
                return RelationTrainingResult(
                    checkpoint_dir=Path(self.config.output_dir),
                    onnx_path=Path(),
                    quantized_path=Path(),
                    metrics=final_metrics,
                )

        if val_loader:
            final_metrics["best_macro_f1"] = best_f1
            self._ensure_metric_threshold(best_f1)

        checkpoint_dir = run_dir / "checkpoint"
        model_dir = checkpoint_dir / "model"
        tokenizer_dir = checkpoint_dir / "tokenizer"
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self._unwrap_model(model)
        model_to_save.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)
        (checkpoint_dir / "labels.json").write_text(json.dumps(label2id, indent=2))

        # Generate calibration parameters if validation set exists
        if val_loader:
            logger.info("Computing calibration parameters from validation set...")
            id2label = {v: k for k, v in label2id.items()}
            calibration = self._compute_calibration(model, val_loader, id2label)
            calibration_path = checkpoint_dir / "calibration.json"
            calibration_path.write_text(json.dumps(calibration, indent=2))
            logger.info(f"Calibration saved to {calibration_path}")
        else:
            logger.warning("No validation set provided; skipping calibration generation.")

        onnx_dir = run_dir / "onnx"
        onnx_path, quantized_path = export_relation_classifier(
            model_dir,
            self.tokenizer,
            onnx_dir,
            self.config.export_opset,
            self.config.max_seq_length,
            self.config.quantize,
        )
        save_training_state(checkpoint_dir / "state.pt", model, optimizer, scheduler, self.ctx.scaler)

        metadata = build_metadata(
            component="relation",
            version=run_dir.name,
            checkpoint_dir=checkpoint_dir,
            onnx_path=onnx_path,
            quantized_path=quantized_path,
            metrics=final_metrics,
            config=self.config.model_dump(mode="json"),
        )
        save_artifact_metadata(metadata, Path(self.artifact_root))
        prune_run_directories(component_dir, self.config.max_checkpoints)

        self.ctx.cleanup()
        return RelationTrainingResult(
            checkpoint_dir=checkpoint_dir,
            onnx_path=onnx_path,
            quantized_path=quantized_path,
            metrics=final_metrics,
        )

    def _evaluate(self, model, loader: DataLoader) -> Dict[str, float]:
        logger.info("Running validation on {} batches", len(loader))
        model.eval()
        preds = []
        labels = []
        
        val_progress = ProgressTracker(
            total=len(loader),
            stage_name="Validation",
            total_stages=1,
            current_stage=1,
            log_interval=self.config.progress_log_interval,
        )
        
        with torch.no_grad():
            for batch in loader:
                labels.extend(batch["labels"].tolist())
                batch = {k: v.to(self.ctx.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits.detach().cpu()
                preds.extend(torch.argmax(logits, dim=-1).tolist())
                val_progress.update()
        
        metrics = {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
        }
        logger.info("Validation complete | Accuracy: {:.4f} | F1: {:.4f}", metrics["accuracy"], metrics["macro_f1"])
        return metrics

    def _compute_calibration(self, model, loader: DataLoader, id2label: Dict[int, str]) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class calibration parameters using temperature scaling.
        
        For each relation type, we analyze the model's confidence distribution
        vs. actual accuracy and compute scale/bias adjustments.
        
        Returns:
            Dictionary mapping label names to {"scale": float, "bias": float}
        """
        model.eval()
        all_logits = []
        all_labels = []
        
        # Collect all predictions and labels
        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"]
                all_labels.append(labels)
                batch = {k: v.to(self.ctx.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits.detach().cpu()
                all_logits.append(logits)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Convert logits to probabilities
        probs = torch.softmax(all_logits, dim=-1)
        pred_labels = torch.argmax(all_logits, dim=-1)
        
        # Compute per-class calibration
        calibration = {}
        num_classes = all_logits.size(1)
        
        for class_id in range(num_classes):
            label_name = id2label.get(class_id, str(class_id))
            
            # Find predictions for this class
            class_mask = pred_labels == class_id
            if class_mask.sum() == 0:
                # No predictions for this class, use neutral calibration
                calibration[label_name] = {"scale": 1.0, "bias": 0.0}
                continue
            
            # Get confidence scores for this class
            class_confidences = probs[class_mask, class_id]
            class_correct = (all_labels[class_mask] == class_id).float()
            
            # Compute average confidence and accuracy
            avg_confidence = class_confidences.mean().item()
            avg_accuracy = class_correct.mean().item()
            
            # Simple linear calibration: we want calibrated_conf ≈ accuracy
            # calibrated = scale * raw_confidence + bias
            # We target: scale * avg_confidence + bias = avg_accuracy
            # Use a simple heuristic: adjust scale to match accuracy
            
            if avg_confidence > 0.0:
                # Scale adjusts the confidence to match empirical accuracy
                scale = avg_accuracy / max(avg_confidence, 0.01)
                # Clamp scale to reasonable range [0.5, 1.5]
                scale = max(0.5, min(1.5, scale))
                
                # Bias corrects for systematic over/under-confidence
                bias = avg_accuracy - scale * avg_confidence
                # Clamp bias to [-0.2, 0.2]
                bias = max(-0.2, min(0.2, bias))
            else:
                scale = 1.0
                bias = 0.0
            
            calibration[label_name] = {"scale": float(scale), "bias": float(bias)}
            
            logger.debug(
                f"Calibration for '{label_name}': "
                f"confidence={avg_confidence:.3f}, accuracy={avg_accuracy:.3f}, "
                f"scale={scale:.3f}, bias={bias:.3f}"
            )
        
        return calibration

    def _maybe_resume(self, model, optimizer, scheduler) -> None:
        resume_state = None
        if self.config.resume_from:
            resume_state = Path(self.config.resume_from) / "state.pt"
        if resume_state and resume_state.exists():
            load_training_state(resume_state, model, optimizer, scheduler, self.ctx.scaler)

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    def _ensure_metric_threshold(self, metric_value: float) -> None:
        if self.config.metric_threshold is None:
            return
        if metric_value < self.config.metric_threshold:
            raise RuntimeError(
                f"Relation validation metric {metric_value:.4f} did not meet threshold "
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


@contextmanager
def _suppress_hf_mismatch_warning():
    logger = hf_logging.get_logger("transformers.modeling_utils")
    previous = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


