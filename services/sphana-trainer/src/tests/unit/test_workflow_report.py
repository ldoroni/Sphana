"""Tests for workflow report generation."""

import json
from pathlib import Path

from sphana_trainer.utils.metadata import build_metadata, save_artifact_metadata
from sphana_trainer.workflow.report import generate_workflow_report


def test_generate_workflow_report(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    state_path = artifact_root / "workflow-state.json"
    state_path.write_text('{"stages": {"ingest": {"timestamp": "now", "output": "artifacts/ingest"}}}')
    component_dir = tmp_path / "run"
    component_dir.mkdir()
    metadata = build_metadata(
        component="embedding",
        version="v1",
        checkpoint_dir=component_dir,
        onnx_path=component_dir / "model.onnx",
        quantized_path=component_dir / "model.int8.onnx",
        metrics={"val_cosine": 0.9},
        config={"model_name": "stub"},
    )
    save_artifact_metadata(metadata, artifact_root)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"components": []}')
    report_path = generate_workflow_report(state_path, artifact_root, manifest)
    report = report_path.read_text()
    assert "embedding" in report
    assert "val_cosine" in report


def test_generate_workflow_report_handles_invalid_manifest(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    state_path = artifact_root / "workflow-state.json"
    state_path.write_text('{"stages": {}}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{bad json")
    report_path = generate_workflow_report(state_path, artifact_root, manifest)
    data = report_path.read_text()
    assert "manifest" in data


def test_generate_workflow_report_includes_telemetry(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    state_path = artifact_root / "workflow-state.json"
    state_path.write_text('{"stages": {}}')
    run_dir = tmp_path / "runs" / "embedding-v1"
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"component": "embedding", "metadata": {"stage": "train", "cpu_percent": 10.0}}) + "\n"
    )
    metadata = build_metadata(
        component="embedding",
        version="v1",
        checkpoint_dir=checkpoint_dir,
        onnx_path=run_dir / "model.onnx",
        quantized_path=run_dir / "model.int8.onnx",
        metrics={"train_loss": 0.1},
        config={"model_name": "stub"},
    )
    save_artifact_metadata(metadata, artifact_root)
    report_path = generate_workflow_report(state_path, artifact_root, None)
    report = json.loads(report_path.read_text())
    assert "telemetry" in report["components"]["embedding"]
    assert report["components"]["embedding"]["telemetry"]["cpu_percent"] == 10.0


def test_generate_workflow_report_skips_invalid_metrics(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    state_path = artifact_root / "workflow-state.json"
    state_path.write_text('{"stages": {}}')
    run_dir = tmp_path / "runs" / "embedding-v2"
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("\nnot-json\n")
    metadata = build_metadata(
        component="embedding",
        version="v2",
        checkpoint_dir=checkpoint_dir,
        onnx_path=run_dir / "model.onnx",
        quantized_path=run_dir / "model.int8.onnx",
        metrics={"train_loss": 0.2},
        config={"model_name": "stub"},
    )
    save_artifact_metadata(metadata, artifact_root)
    report_path = generate_workflow_report(state_path, artifact_root, None)
    report = json.loads(report_path.read_text())
    assert "telemetry" not in report["components"]["embedding"]

