"""Tests for artifact core helpers."""

import json
from pathlib import Path

import pytest

from sphana_trainer.artifacts import core
from sphana_trainer.utils.metadata import build_metadata, save_artifact_metadata


def _write_metadata(artifact_root: Path, component: str, version: str):
    checkpoint_dir = artifact_root / component / version / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.onnx"
    model_path.write_text("onnx")
    meta = build_metadata(
        component=component,
        version=version,
        checkpoint_dir=checkpoint_dir,
        onnx_path=model_path,
        quantized_path=model_path,
        metrics={"val": 0.9},
        config={"model_name": "stub"},
    )
    save_artifact_metadata(meta, artifact_root)
    return meta


def test_list_show_diff_promote(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    meta1 = _write_metadata(artifact_root, "embedding", "v1")
    meta2 = _write_metadata(artifact_root, "embedding", "v2")

    listings = core.list_artifacts(artifact_root)
    assert "embedding" in listings
    assert len(listings["embedding"]) == 2

    shown = core.show_artifact("embedding", "v1", artifact_root)
    assert shown.version == "v1"

    diff = core.diff_artifacts(meta1, meta2)
    assert "val" in diff

    latest = core.promote_artifact("embedding", "v2", artifact_root)
    assert latest.exists()
    pointer = artifact_root / "embedding" / "latest.json"
    assert json.loads(pointer.read_text())["metadata"] == str(latest)


def test_promote_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.promote_artifact("missing", "v1", tmp_path)


def test_show_artifact_missing_entry(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.show_artifact("embedding", "v1", tmp_path)


def test_history_loader_handles_blanks(tmp_path):
    component_dir = tmp_path / "embedding"
    component_dir.mkdir(parents=True, exist_ok=True)
    history = component_dir / "history.jsonl"
    history.write_text("\n{}\n")
    entries = core._load_history_entries("embedding", tmp_path)
    assert len(entries) == 1


def test_load_metadata_missing_file(tmp_path):
    bogus = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        core._load_metadata(bogus)


def test_list_artifacts_handles_empty_history(tmp_path):
    artifact_root = tmp_path / "artifacts"
    history = artifact_root / "embedding" / "history.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text("\n")
    entries = core.list_artifacts(artifact_root)
    assert entries["embedding"] == []

