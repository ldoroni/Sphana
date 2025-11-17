"""Workflow reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from sphana_trainer.utils.metadata import load_artifact_metadata
from sphana_trainer.workflow.state import load_workflow_state


def generate_workflow_report(
    state_path: Path,
    artifact_root: Path,
    manifest_path: Optional[Path],
) -> Path:
    artifact_root = artifact_root.expanduser().resolve()
    state = load_workflow_state(state_path)
    report = {"stages": state.get("stages", {})}

    components = {}
    for component in ("embedding", "relation", "gnn"):
        try:
            meta = load_artifact_metadata(component, artifact_root)
        except FileNotFoundError:
            continue
        components[component] = {
            "version": meta.version,
            "metrics": meta.metrics,
            "onnx_path": meta.onnx_path,
            "timestamp": meta.timestamp,
        }
        telemetry = _load_telemetry_snapshot(Path(meta.checkpoint_dir))
        if telemetry:
            components[component]["telemetry"] = telemetry
    report["components"] = components

    if manifest_path and manifest_path.exists():
        try:
            report["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["manifest"] = {"path": str(manifest_path)}

    output = artifact_root / "workflow-report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return output


def _load_telemetry_snapshot(checkpoint_dir: Path) -> Optional[dict]:
    run_dir = Path(checkpoint_dir).parent
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    last_record = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last_record = json.loads(line)
            except json.JSONDecodeError:
                continue
    if not last_record:
        return None
    metadata = last_record.get("metadata") or {}
    snapshot = {k: v for k, v in metadata.items() if k != "stage"}
    return snapshot or None

