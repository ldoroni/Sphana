"""Simple structured metrics logger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class MetricsLogger:
    """Append-only JSONL metrics writer."""

    def __init__(self, run_dir: Path, component: str, run_id: str) -> None:
        self.run_dir = run_dir
        self.component = component
        self.run_id = run_id
        self.path = run_dir / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": self.component,
            "run_id": self.run_id,
            "step": step,
            "metrics": metrics,
            "metadata": metadata or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

