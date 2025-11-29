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
        logger.info("=" * 80)
        logger.info("STARTING: Embedding training task")
        logger.info("Config: model={}, dataset={}, epochs={}, batch_size={}", 
                    self.config.model_name, 
                    self.config.dataset_path,
                    self.config.epochs,
                    self.config.batch_size)
        logger.info("=" * 80)
        
        trainer = EmbeddingTrainer(self.config, self.artifact_root)
        result = trainer.train()
        
        logger.success("=" * 80)
        logger.success("COMPLETED: Embedding training task")
        logger.success("Results: checkpoint={}, metrics={}", 
                      result.checkpoint_dir, 
                      result.metrics)
        logger.success("=" * 80)
        return result




