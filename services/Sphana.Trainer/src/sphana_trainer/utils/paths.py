"""Filesystem helpers."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def resolve_split_file(
    base_path: Path,
    override: Optional[Path],
    split_name: str,
) -> Optional[Path]:
    """Resolve a dataset split path based on overrides and conventions."""

    if override:
        return override

    if base_path.is_file():
        return base_path if split_name == "train" else None

    candidates = [
        base_path / f"{split_name}.jsonl",
        base_path / f"{split_name}.json",
        base_path / f"{split_name}.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def prepare_run_directory(component_dir: Path, version: Optional[str] = None) -> Path:
    """Create a timestamped directory for the current training run."""

    slug = version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = component_dir / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prune_run_directories(component_dir: Path, keep: int) -> None:
    if keep <= 0 or not component_dir.exists():
        return
    runs = [path for path in component_dir.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for obsolete in runs[keep:]:
        shutil.rmtree(obsolete, ignore_errors=True)


