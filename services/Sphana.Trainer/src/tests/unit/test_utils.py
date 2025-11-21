"""Unit tests for utility helpers."""

import json

import pytest

from sphana_trainer.utils import (
    build_metadata,
    load_artifact_metadata,
    prepare_run_directory,
    resolve_split_file,
    save_artifact_metadata,
)
from sphana_trainer.utils.fingerprint import dataset_fingerprint
from sphana_trainer.utils.metrics import MetricsLogger


def test_resolve_split_file_prefers_override(tmp_path):
    base = tmp_path / "dataset"
    base.mkdir()
    (base / "train.jsonl").write_text('{"query": "a", "positive": "b"}\n')
    override = tmp_path / "custom.jsonl"
    override.write_text("{}")

    resolved = resolve_split_file(base, override, "train")
    assert resolved == override


def test_resolve_split_file_directory_and_missing(tmp_path):
    split_file = tmp_path / "dir" / "validation.json"
    split_file.parent.mkdir()
    split_file.write_text("{}")
    resolved_val = resolve_split_file(split_file.parent, None, "validation")
    assert resolved_val == split_file

    resolved_none = resolve_split_file(split_file, None, "validation")
    assert resolved_none is None

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert resolve_split_file(empty_dir, None, "train") is None


def test_prepare_run_directory_uses_version(tmp_path):
    component_dir = tmp_path / "embedding"
    run_dir = prepare_run_directory(component_dir, version="1.0.0")
    assert run_dir.exists()
    assert run_dir.name == "1.0.0"


def test_metadata_roundtrip(tmp_path):
    artifact_root = tmp_path / "artifacts"
    checkpoint = tmp_path / "artifacts" / "embedding" / "1.0.0"
    checkpoint.mkdir(parents=True)
    onnx = checkpoint / "embedding.onnx"
    onnx.write_text("onnx")
    metadata = build_metadata(
        component="embedding",
        version="1.0.0",
        checkpoint_dir=checkpoint,
        onnx_path=onnx,
        quantized_path=onnx,
        metrics={"loss": 0.1},
        config={"k": "v"},
    )
    save_artifact_metadata(metadata, artifact_root)
    loaded = load_artifact_metadata("embedding", artifact_root)
    assert loaded.component == "embedding"
    assert loaded.metrics["loss"] == 0.1


def test_metadata_missing_files(tmp_path):
    artifact_root = tmp_path / "artifacts"
    with pytest.raises(FileNotFoundError):
        load_artifact_metadata("missing", artifact_root)

    pointer = artifact_root / "embedding" / "latest.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"metadata": str(tmp_path / "does_not_exist.json")}))
    with pytest.raises(FileNotFoundError):
        load_artifact_metadata("embedding", artifact_root)


def test_dataset_fingerprint_changes_when_file_changes(tmp_path):
    dataset_file = tmp_path / "data.jsonl"
    dataset_file.write_text("hello\n")
    fp1 = dataset_fingerprint(dataset_file)
    dataset_file.write_text("hello world\n")
    fp2 = dataset_fingerprint(dataset_file)
    assert fp1 != fp2


def test_metrics_logger_writes_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logger = MetricsLogger(run_dir, "embedding", "run-1")
    logger.log(0, {"loss": 0.5}, {"stage": "train", "device": "cpu"})
    contents = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(contents) == 1
    record = json.loads(contents[0])
    assert record["metrics"]["loss"] == 0.5
    assert record["metadata"]["stage"] == "train"
    assert record["metadata"]["device"] == "cpu"

