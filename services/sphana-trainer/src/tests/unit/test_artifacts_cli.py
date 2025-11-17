"""Tests for artifact helper utilities."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sphana_trainer.artifacts import diff_artifacts, list_artifacts, promote_artifact, show_artifact
from sphana_trainer.artifacts.publisher import publish_manifest
from sphana_trainer.cli import app
from sphana_trainer.utils import build_metadata, save_artifact_metadata


def _write_metadata(tmp_path: Path, component: str, version: str) -> Path:
    artifact_root = tmp_path / "artifacts"
    checkpoint_dir = artifact_root / component / version / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = checkpoint_dir / "model.onnx"
    onnx_path.write_text("onnx")
    metadata = build_metadata(
        component=component,
        version=version,
        checkpoint_dir=checkpoint_dir,
        onnx_path=onnx_path,
        quantized_path=onnx_path,
        metrics={"loss": float(len(version))},
        config={"version": version},
    )
    save_artifact_metadata(metadata, artifact_root)
    return artifact_root


def test_list_and_show_artifacts(tmp_path):
    artifact_root = _write_metadata(tmp_path, "embedding", "v1")
    entries = list_artifacts(artifact_root, "embedding")
    assert "embedding" in entries
    assert entries["embedding"][0].version == "v1"

    meta = show_artifact("embedding", "v1", artifact_root)
    assert meta.metrics["loss"] == float(len("v1"))


def test_diff_and_promote(tmp_path):
    artifact_root = _write_metadata(tmp_path, "relation", "v1")
    _write_metadata(tmp_path, "relation", "v2")
    meta_a = show_artifact("relation", "v1", artifact_root)
    meta_b = show_artifact("relation", "v2", artifact_root)
    diff = diff_artifacts(meta_a, meta_b)
    assert "loss" in diff

    latest_pointer = promote_artifact("relation", "v2", artifact_root)
    pointer = json.loads((artifact_root / "relation" / "latest.json").read_text())
    assert pointer["metadata"] == str(latest_pointer)


def test_publish_manifest_posts(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    class FakeResponse:
        def __init__(self):
            self.text = "ok"

        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        assert "manifest" in json
        assert timeout == 10
        return FakeResponse()

    monkeypatch.setattr("sphana_trainer.artifacts.publisher.requests.post", fake_post)
    assert publish_manifest(manifest, "http://example") == "ok"

    with pytest.raises(FileNotFoundError):
        publish_manifest(manifest.parent / "missing.json", "http://example")


def test_artifacts_bundle(monkeypatch, tmp_path):
    runner = CliRunner()

    class DummyMeta:
        checkpoint_dir = str(tmp_path / "ckpt")
        onnx_path = str(tmp_path / "model.onnx")
        quantized_path = str(tmp_path / "model.int8.onnx")

    (tmp_path / "ckpt").mkdir()
    (tmp_path / "model.onnx").write_text("onnx")
    (tmp_path / "model.int8.onnx").write_text("onnx")
    (tmp_path / "ckpt" / "metadata.json").write_text("{}")

    monkeypatch.setattr("sphana_trainer.cli.show_artifact", lambda *a, **k: DummyMeta())

    result = runner.invoke(
        app,
        [
            "artifacts",
            "bundle",
            "embedding",
            "v1",
            str(tmp_path / "bundle"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "bundle" / "model.onnx").exists()
    assert (tmp_path / "bundle" / "model.int8.onnx").exists()


def test_artifacts_bundle_handles_missing_paths(monkeypatch, tmp_path):
    runner = CliRunner()

    class DummyMeta:
        checkpoint_dir = str(tmp_path / "ckpt")
        onnx_path = ""
        quantized_path = ""

    monkeypatch.setattr("sphana_trainer.cli.show_artifact", lambda *a, **k: DummyMeta())
    result = runner.invoke(
        app,
        [
            "artifacts",
            "bundle",
            "embedding",
            "v1",
            str(tmp_path / "bundle2"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ],
    )
    assert result.exit_code == 0


def test_artifacts_list_show_diff_promote_cli(monkeypatch, tmp_path):
    runner = CliRunner()
    entries = {"embedding": [SimpleNamespace(version="v1", onnx_path="model.onnx")]}
    monkeypatch.setattr("sphana_trainer.cli.list_artifacts", lambda *a, **k: entries)
    result = runner.invoke(app, ["artifacts", "list"])
    assert result.exit_code == 0

    show_meta = SimpleNamespace(
        version="v1",
        metrics={"loss": 0.1},
        onnx_path="model.onnx",
        quantized_path="",
        checkpoint_dir=str(tmp_path),
    )
    monkeypatch.setattr("sphana_trainer.cli.show_artifact", lambda *a, **k: show_meta)
    result = runner.invoke(app, ["artifacts", "show", "embedding", "v1"])
    assert result.exit_code == 0

    monkeypatch.setattr("sphana_trainer.cli.diff_artifacts", lambda *a, **k: {"loss": {"v1": 0.1}})
    result = runner.invoke(app, ["artifacts", "diff", "embedding", "v1", "v2"])
    assert result.exit_code == 0

    promoted = tmp_path / "embedding" / "v1" / "metadata.json"
    promoted.parent.mkdir(parents=True, exist_ok=True)
    promoted.write_text("{}")

    monkeypatch.setattr("sphana_trainer.cli.promote_artifact", lambda *a, **k: promoted)
    updated = {}
    published = {}
    monkeypatch.setattr("sphana_trainer.cli._update_manifest", lambda manifest, comp, meta: updated.setdefault(comp, manifest))
    monkeypatch.setattr("sphana_trainer.cli.publish_manifest", lambda manifest, url: published.setdefault(url, manifest))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    result = runner.invoke(
        app,
        [
            "artifacts",
            "promote",
            "embedding",
            "v1",
            "--manifest",
            str(manifest),
            "--publish-url",
            "http://example",
            "--artifact-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert str(updated["embedding"]) == str(manifest)
    assert str(published["http://example"]) == str(manifest)

    result = runner.invoke(
        app,
        [
            "artifacts",
            "promote",
            "embedding",
            "v1",
            "--publish-url",
            "http://example",
        ],
    )
    assert result.exit_code != 0

