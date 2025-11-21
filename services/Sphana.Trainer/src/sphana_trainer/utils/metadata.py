"""Artifact metadata helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class ArtifactMetadata:
    component: str
    version: str
    checkpoint_dir: str
    onnx_path: str
    quantized_path: str
    metrics: Dict[str, float]
    config: Dict[str, Any]
    timestamp: str


def save_artifact_metadata(metadata: ArtifactMetadata, artifact_root: Path) -> Path:
    """Persist run metadata and update the latest pointer."""

    record = asdict(metadata)
    metadata_path = Path(metadata.checkpoint_dir) / "metadata.json"
    metadata_path.write_text(json.dumps(record, indent=2))

    latest_dir = artifact_root / metadata.component
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "latest.json").write_text(json.dumps({"metadata": str(metadata_path)}))
    history_path = latest_dir / "history.jsonl"
    _append_history(history_path, record["version"], metadata_path)
    return metadata_path


def load_artifact_metadata(component: str, artifact_root: Path) -> ArtifactMetadata:
    pointer = artifact_root / component / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError(f"No metadata found for component '{component}' in {artifact_root}")
    with pointer.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata_path = Path(payload["metadata"])
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file missing at {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ArtifactMetadata(**data)


def build_metadata(
    component: str,
    version: str,
    checkpoint_dir: Path,
    onnx_path: Path,
    quantized_path: Path,
    metrics: Dict[str, float],
    config: Dict[str, Any],
) -> ArtifactMetadata:
    return ArtifactMetadata(
        component=component,
        version=version,
        checkpoint_dir=str(checkpoint_dir),
        onnx_path=str(onnx_path),
        quantized_path=str(quantized_path),
        metrics=metrics,
        config=config,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _append_history(history_path: Path, version: str, metadata_path: Path) -> None:
    entries = []
    if history_path.exists():
        entries = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    entries = [entry for entry in entries if entry.get("version") != version]
    entries.append({"version": version, "metadata": str(metadata_path)})
    entries.sort(key=lambda item: item["version"])
    with history_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


