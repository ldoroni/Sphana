"""Logging helpers for the trainer CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loguru import logger


def init_logging(log_file: Optional[Path] = None, level: str = "INFO") -> None:
    """Configure loguru for console + optional file logging."""

    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        level=level.upper(),
        enqueue=True,
    )

    target = log_file or _default_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.add(target, level=level.upper(), rotation="10 MB", retention="7 days")


def _default_log_path() -> Path:
    workspace = os.environ.get("SPHANA_WORKSPACE", "target")
    return Path(workspace).expanduser().resolve() / "logs" / "trainer.log"


