"""Embedding model training workflow."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sphana_trainer.training import EmbeddingTrainer
from sphana_trainer.tasks.base import BaseTask


class EmbeddingTask(BaseTask):
    """Train and export the embedding encoder."""

    def __init__(self, config, artifact_root):
        super().__init__(config)
        self.artifact_root = Path(artifact_root)

    def run(self):
        logger.info("Starting embedding training with model {}", self.config.model_name)
        trainer = EmbeddingTrainer(self.config, self.artifact_root)
        result = trainer.train()
        logger.success("Embedding artifacts saved to {}", result.checkpoint_dir)
        return result



