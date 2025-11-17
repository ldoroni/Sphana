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
        logger.info(
            "Training GNN ranker with hidden_dim={} and layers={}",
            self.config.hidden_dim,
            self.config.num_layers,
        )
        trainer = GNNTrainer(self.config, self.artifact_root)
        result = trainer.train()
        logger.success("GNN artifacts saved to {}", result.checkpoint_dir)
        return result



