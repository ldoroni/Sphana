"""Workflow state helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional


def load_workflow_state(path: Path) -> Dict:
    path = path.expanduser().resolve()
    if not path.exists():
        return {"stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: Dict) -> Dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return state


def stage_is_current(state: Dict, stage: str, output_path: Path | None) -> bool:
    record = state.get("stages", {}).get(stage)
    if not record or record.get("status") != "succeeded":
        return False
    if output_path:
        return output_path.exists()
    return True


def record_stage_start(path: Path, stage: str) -> Dict:
    state = load_workflow_state(path)
    entry = state.setdefault("stages", {}).setdefault(stage, {})
    entry.update(
        {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }
    )
    return _write_state(path, state)


def record_stage_success(path: Path, stage: str, output_path: Path | None) -> Dict:
    state = load_workflow_state(path)
    entry = state.setdefault("stages", {}).setdefault(stage, {})
    entry.update(
        {
            "status": "succeeded",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "output": str(output_path) if output_path else None,
            "error": None,
        }
    )
    return _write_state(path, state)


def record_stage_failure(path: Path, stage: str, error: Exception | str) -> Dict:
    state = load_workflow_state(path)
    entry = state.setdefault("stages", {}).setdefault(stage, {})
    entry.update(
        {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(error),
        }
    )
    return _write_state(path, state)


class WorkflowLock:
    """File-based lock to prevent concurrent workflow runs."""

    def __init__(self, path: Path, stale_minutes: int = 30) -> None:
        self.path = path.expanduser().resolve()
        self.stale = timedelta(minutes=stale_minutes)
        self._acquired = False

    def acquire(self, *, force: bool = False) -> None:
        now = datetime.now(timezone.utc)
        if self.path.exists():
            try:
                info = json.loads(self.path.read_text(encoding="utf-8"))
                timestamp = datetime.fromisoformat(info.get("timestamp"))
            except Exception:
                timestamp = None
            if not force and timestamp and now - timestamp < self.stale:
                raise RuntimeError(
                    f"Workflow already running (lock: {self.path}). Use --force-lock to override."
                )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": now.isoformat(), "pid": os.getpid()}
        self.path.write_text(json.dumps(payload, indent=2))
        self._acquired = True

    def release(self) -> None:
        if self._acquired and self.path.exists():
            try:
                self.path.unlink()
            finally:
                self._acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def clear_stage(path: Path, stage: str) -> Dict:
    state = load_workflow_state(path)
    if stage in state.get("stages", {}):
        del state["stages"][stage]
    return _write_state(path, state)

