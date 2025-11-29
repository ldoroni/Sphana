"""GNN ranking model training workflow."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sphana_trainer.training import GNNTrainer
from sphana_trainer.tasks.base import BaseTask


class GNNTask(BaseTask):
    """Train and export the GGNN ranker."""

    def __init__(self, config, artifact_root):
        super().__init__(config)
        self.artifact_root = Path(artifact_root)

    def run(self):
        logger.info("=" * 80)
        logger.info("STARTING: GNN ranking training task")
        logger.info("Config: hidden_dim={}, layers={}, dataset={}, epochs={}", 
                    self.config.hidden_dim,
                    self.config.num_layers,
                    self.config.dataset_path,
                    self.config.epochs)
        logger.info("=" * 80)
        
        trainer = GNNTrainer(self.config, self.artifact_root)
        result = trainer.train()
        
        logger.success("=" * 80)
        logger.success("COMPLETED: GNN ranking training task")
        logger.success("Results: checkpoint={}, metrics={}", 
                      result.checkpoint_dir, 
                      result.metrics)
        logger.success("=" * 80)
        return result




