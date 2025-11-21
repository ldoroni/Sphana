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
        logger.info("Training relation extractor with base model {}", self.config.model_name)
        trainer = RelationExtractionTrainer(self.config, self.artifact_root)
        result = trainer.train()
        logger.success("Relation artifacts saved to {}", result.checkpoint_dir)
        return result



