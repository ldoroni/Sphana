"""Ingestion task wiring for CLI."""

from __future__ import annotations

from loguru import logger

from sphana_trainer.data.pipeline import IngestionConfig, run_ingestion
from sphana_trainer.tasks.base import BaseTask


class IngestionTask(BaseTask):
    def __init__(self, config: IngestionConfig, force: bool = False):
        super().__init__(config)
        self.force = force

    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("STARTING: Ingestion task")
        logger.info("Config: parser={}, chunk_size={}, relation_model={}", 
                    self.config.parser, 
                    self.config.chunk_size,
                    self.config.relation_model or "none")
        logger.info("=" * 80)
        
        result = run_ingestion(self.config, force=self.force)
        
        logger.success("=" * 80)
        logger.success("COMPLETED: Ingestion task")
        logger.success("Results: docs={}, chunks={}, relations={}",
            result.document_count,
            result.chunk_count,
            result.relation_count)
        logger.success("Chunks output: {}", result.chunks_output_dir)
        logger.success("Relations output: {}", result.relations_output_dir)
        logger.success("=" * 80)


