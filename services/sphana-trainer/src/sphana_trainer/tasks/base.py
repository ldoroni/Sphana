"""Shared base task for trainer workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


class BaseTask(ABC):
    """Base class for concrete training/export tasks."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.console = Console()

    @abstractmethod
    def run(self) -> None:
        """Execute the task."""

    def _simulate_stage(self, stage: str, message: str) -> None:
        """Emit a log and progress placeholder."""

        logger.info(f"[{stage}] {message}")
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            transient=True,
            console=self.console,
        ) as progress:
            progress.add_task(description=message, total=None)

    def _prepare_output_dir(self) -> Path:
        target = Path(self.config.output_dir)
        timestamped = target / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        timestamped.mkdir(parents=True, exist_ok=True)
        logger.debug("Prepared output directory {}", timestamped)
        return timestamped


