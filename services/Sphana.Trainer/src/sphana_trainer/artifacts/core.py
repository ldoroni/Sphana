"""Helpers for inspecting and promoting trainer artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from sphana_trainer.utils.metadata import ArtifactMetadata


def list_artifacts(artifact_root: Path, component: Optional[str] = None) -> Dict[str, List[ArtifactMetadata]]:
    """Return known artifacts per component."""

    artifact_root = artifact_root.expanduser().resolve()
    components = [component] if component else sorted(
        {path.name for path in artifact_root.iterdir() if path.is_dir()}
    )
    results: Dict[str, List[ArtifactMetadata]] = {}
    for comp in components:
        entries = []
        for entry in _load_history_entries(comp, artifact_root):
            metadata = _load_metadata(Path(entry["metadata"]))
            entries.append(metadata)
        results[comp] = entries
    return results


def show_artifact(component: str, version: str, artifact_root: Path) -> ArtifactMetadata:
    entry = _find_history_entry(component, version, artifact_root)
    if not entry:
        raise FileNotFoundError(f"No artifact for component '{component}' and version '{version}'")
    return _load_metadata(Path(entry["metadata"]))


def diff_artifacts(meta_a: ArtifactMetadata, meta_b: ArtifactMetadata) -> Dict[str, Dict[str, float]]:
    metrics_diff = {}
    for key in set(meta_a.metrics) | set(meta_b.metrics):
        metrics_diff[key] = {
            meta_a.version: meta_a.metrics.get(key),
            meta_b.version: meta_b.metrics.get(key),
        }
    return metrics_diff


def promote_artifact(component: str, version: str, artifact_root: Path) -> Path:
    entry = _find_history_entry(component, version, artifact_root)
    if not entry:
        raise FileNotFoundError(f"No artifact for component '{component}' and version '{version}'")
    metadata_path = Path(entry["metadata"])
    pointer = artifact_root / component / "latest.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"metadata": str(metadata_path)}))
    return metadata_path


def _load_history_entries(component: str, artifact_root: Path) -> List[Dict[str, str]]:
    history_file = artifact_root / component / "history.jsonl"
    if not history_file.exists():
        return []
    entries = []
    with history_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _find_history_entry(component: str, version: str, artifact_root: Path) -> Optional[Dict[str, str]]:
    for entry in _load_history_entries(component, artifact_root):
        if entry.get("version") == version:
            return entry
    return None


def _load_metadata(path: Path) -> ArtifactMetadata:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file missing at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ArtifactMetadata(**data)

