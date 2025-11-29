"""Relation extraction training workflow."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sphana_trainer.training import RelationExtractionTrainer
from sphana_trainer.tasks.base import BaseTask


class RelationExtractionTask(BaseTask):
    """Train and export the relation extraction model."""

    def __init__(self, config, artifact_root):
        super().__init__(config)
        self.artifact_root = Path(artifact_root)

    def run(self):
        logger.info("=" * 80)
        logger.info("STARTING: Relation extraction training task")
        logger.info("Config: model={}, dataset={}, epochs={}, batch_size={}", 
                    self.config.model_name, 
                    self.config.dataset_path,
                    self.config.epochs,
                    self.config.batch_size)
        logger.info("=" * 80)
        
        trainer = RelationExtractionTrainer(self.config, self.artifact_root)
        result = trainer.train()
        
        logger.success("=" * 80)
        logger.success("COMPLETED: Relation extraction training task")
        logger.success("Results: checkpoint={}, metrics={}", 
                      result.checkpoint_dir, 
                      result.metrics)
        logger.success("=" * 80)
        return result




