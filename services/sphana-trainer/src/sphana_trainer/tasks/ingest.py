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
        logger.info("Starting ingestion pipeline")
        result = run_ingestion(self.config, force=self.force)
        logger.success(
            "Ingestion complete: docs={}, chunks={}, relations={}, output={}",
            result.document_count,
            result.chunk_count,
            result.relation_count,
            result.output_dir,
        )

