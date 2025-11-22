"""NER export task."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

from loguru import logger
from transformers import AutoTokenizer

from sphana_trainer.config import NerConfig
from sphana_trainer.exporters.onnx import export_ner_model
from sphana_trainer.tasks.base import BaseTask
from sphana_trainer.utils import (
    build_metadata,
    prepare_run_directory,
    prune_run_directories,
    save_artifact_metadata,
)


@dataclass
class NerTrainingResult:
    checkpoint_dir: Path
    onnx_path: Path
    quantized_path: Path
    metrics: Dict[str, float]


class NerTask(BaseTask):
    """Export NER model to ONNX."""

    def __init__(self, config: NerConfig, artifact_root: Path):
        self.config = config
        self.artifact_root = Path(artifact_root)

    def run(self) -> NerTrainingResult:
        component_dir = Path(self.config.output_dir)
        run_dir = prepare_run_directory(component_dir, self.config.version)
        
        logger.info("Exporting NER model {}...", self.config.model_name)
        
        # We are not training, just exporting the base model
        # In a real scenario, we would fine-tune here if dataset_path was provided
        
        onnx_dir = run_dir / "onnx"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        onnx_path, quantized_path = export_ner_model(
            self.config.model_name,
            tokenizer,
            onnx_dir,
            self.config.export_opset,
            self.config.max_seq_length,
            self.config.quantize,
            allowed_mismatch_ratio=self.config.quantization_mismatch_threshold,
        )
        
        # Save tokenizer for reproducibility
        checkpoint_dir = run_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(checkpoint_dir)
        
        metrics = {} # No training metrics
        
        metadata = build_metadata(
            component="ner",
            version=run_dir.name,
            checkpoint_dir=checkpoint_dir,
            onnx_path=onnx_path,
            quantized_path=quantized_path,
            metrics=metrics,
            config=self.config.model_dump(mode="json"),
        )
        save_artifact_metadata(metadata, self.artifact_root)
        prune_run_directories(component_dir, self.config.max_checkpoints)

        logger.success("NER artifacts saved to {}", run_dir)
        return NerTrainingResult(
            checkpoint_dir=checkpoint_dir,
            onnx_path=onnx_path,
            quantized_path=quantized_path,
            metrics=metrics,
        )

