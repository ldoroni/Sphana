"""End-to-end test that exercises the CLI over sample data."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from sphana_trainer.cli import app  # type: ignore[import]


DATA_DIR = (Path(__file__).resolve().parents[1] / "data").resolve()
TINY_MODEL = "hf-internal-testing/tiny-random-bert"


def _base_config(workspace: Path) -> dict:
    return {
        "workspace_dir": str(workspace),
        "artifact_root": str(workspace / "artifacts"),
    }


def _write_config(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)
    return path


def test_full_pipeline(tmp_path):
    workspace = tmp_path / "workspace"
    embedding_cfg = _base_config(workspace)
    embedding_cfg["embedding"] = {
        "model_name": TINY_MODEL,
        "output_dir": str(workspace / "artifacts" / "embedding"),
        "dataset_path": str(DATA_DIR / "embedding"),
        "train_file": str(DATA_DIR / "embedding" / "train.jsonl"),
        "validation_file": str(DATA_DIR / "embedding" / "validation.jsonl"),
        "version": "test-embedding",
        "batch_size": 2,
        "learning_rate": 5e-4,
        "epochs": 1,
        "max_seq_length": 64,
        "pooling_strategy": "mean",
        "export_opset": 18,
        "quantize": True,
        "temperature": 0.05,
        "eval_subset_size": 2,
    }
    relation_cfg = _base_config(workspace)
    relation_cfg["relation"] = {
        "model_name": TINY_MODEL,
        "output_dir": str(workspace / "artifacts" / "relation"),
        "dataset_path": str(DATA_DIR / "relation"),
        "train_file": str(DATA_DIR / "relation" / "train.jsonl"),
        "validation_file": str(DATA_DIR / "relation" / "validation.jsonl"),
        "version": "test-relation",
        "dependency_cache": str(workspace / "dep-cache"),
        "batch_size": 2,
        "learning_rate": 5e-4,
        "epochs": 1,
        "negative_sampling_ratio": 0.5,
        "export_opset": 18,
        "quantize": True,
        "max_seq_length": 64,
        "eval_batch_size": 2,
        "early_stopping_patience": 1,
    }
    gnn_cfg = _base_config(workspace)
    gnn_cfg["gnn"] = {
        "model_name": "sphana/ggnn-test",
        "output_dir": str(workspace / "artifacts" / "gnn"),
        "dataset_path": str(DATA_DIR / "graphs"),
        "train_file": str(DATA_DIR / "graphs" / "train.jsonl"),
        "validation_file": str(DATA_DIR / "graphs" / "validation.jsonl"),
        "version": "test-gnn",
        "batch_size": 1,
        "learning_rate": 1e-3,
        "epochs": 1,
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.1,
        "listwise_loss": "listnet",
        "export_opset": 18,
        "quantize": True,
        "max_nodes": 16,
        "max_edges": 16,
        "temperature": 1.0,
    }
    export_cfg = _base_config(workspace)
    manifest_path = workspace / "manifests" / "latest.json"
    export_cfg["export"] = {
        "manifest_path": str(manifest_path),
        "include_models": ["embedding", "relation", "gnn"],
        "publish_uri": None,
        "artifact_root": export_cfg["artifact_root"],
    }

    ingest_cfg = {
        "input_dir": str(DATA_DIR / "ingest_sample"),
        "output_dir": str(workspace / "artifacts" / "ingest"),
        "chunk_size": 20,
        "chunk_overlap": 5,
    }

    cfg_dir = tmp_path / "configs"
    ingest_path = _write_config(cfg_dir / "ingest.yaml", ingest_cfg)
    embedding_path = _write_config(cfg_dir / "embedding.yaml", embedding_cfg)
    relation_path = _write_config(cfg_dir / "relation.yaml", relation_cfg)
    gnn_path = _write_config(cfg_dir / "gnn.yaml", gnn_cfg)
    export_path = _write_config(cfg_dir / "export.yaml", export_cfg)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "--ingest-config",
            str(ingest_path),
            "--embedding-config",
            str(embedding_path),
            "--relation-config",
            str(relation_path),
            "--gnn-config",
            str(gnn_path),
            "--export-config",
            str(export_path),
            "--package-config",
            str(export_path),
            "--promote-component",
            "embedding",
            "--promote-version",
            "test-embedding",
            "--manifest",
            str(manifest_path),
            "--artifact-root",
            str(workspace / "artifacts"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["components"]) == {"embedding", "relation", "gnn"}
    for component in manifest["components"]:
        artifact_path = Path(manifest["artifacts"][component])
        assert artifact_path.exists()

    package_path = manifest_path.with_suffix(".tar.gz")
    assert package_path.exists()

    report_path = workspace / "artifacts" / "workflow-report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "embedding" in report["components"]

