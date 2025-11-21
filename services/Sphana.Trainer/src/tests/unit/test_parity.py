"""Tests for ONNX parity helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sphana_trainer.artifacts import parity


class DummyTokenizer:
    def __call__(self, _text, **kwargs):
        max_len = kwargs.get("max_length", 4)
        arr = np.arange(max_len).reshape(1, max_len).astype(np.int64)
        return {"input_ids": arr, "attention_mask": np.ones_like(arr)}


class DummySession:
    def __init__(self, path, *_args, **_kwargs):
        self.path = str(path)

    def run(self, *_args, **_kwargs):
        if "embedding" in self.path:
            return [np.zeros((1, 3), dtype=np.float32)]
        if "relation" in self.path:
            return [np.zeros((1, 2), dtype=np.float32)]
        return [np.array([[0.5]], dtype=np.float32)]


@pytest.fixture(autouse=True)
def _patch_parity_deps(monkeypatch):
    monkeypatch.setattr(
        parity,
        "load_artifact_metadata",
        lambda component, root: SimpleNamespace(
            version="v1",
            onnx_path=str(Path(root) / f"{component}.onnx"),
            quantized_path="",
            config={"model_name": "stub", "max_seq_length": 8},
        ),
    )
    monkeypatch.setattr(parity.AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr(parity.ort, "InferenceSession", DummySession)


def test_generate_parity_sample_embedding(tmp_path):
    sample = tmp_path / "embedding.jsonl"
    sample.write_text(json.dumps({"query": "Explain embeddings."}) + "\n")
    report = parity.generate_parity_sample("embedding", sample, tmp_path)
    assert report["component"] == "embedding"
    assert "embedding" in report["outputs"]


def test_generate_parity_sample_relation(tmp_path):
    sample = tmp_path / "relation.jsonl"
    sample.write_text(
        json.dumps(
            {
                "text": "Alice built NRDB.",
                "entity1": {"text": "Alice", "start": 0, "end": 5},
                "entity2": {"text": "NRDB", "start": 12, "end": 16},
            }
        )
        + "\n"
    )
    report = parity.generate_parity_sample("relation", sample, tmp_path)
    assert report["component"] == "relation"
    assert "probabilities" in report["outputs"]


def test_generate_parity_sample_gnn(tmp_path):
    sample = tmp_path / "gnn.jsonl"
    sample.write_text(
        json.dumps(
            {
                "query_id": "doc-1",
                "candidates": [
                    {
                        "node_features": [[0.1, 0.2], [0.2, 0.3]],
                        "edge_index": [[0, 1]],
                        "edge_directions": [0],
                        "label": 0.9,
                    }
                ],
            }
        )
        + "\n"
    )
    report = parity.generate_parity_sample("gnn", sample, tmp_path)
    assert report["component"] == "gnn"
    assert "score" in report["outputs"]


def test_generate_parity_sample_invalid_component(tmp_path):
    sample = tmp_path / "sample.jsonl"
    sample.write_text("{}\n")
    with pytest.raises(ValueError):
        parity.generate_parity_sample("unknown", sample, tmp_path)


def test_generate_parity_sample_relation_missing_fields(tmp_path):
    sample = tmp_path / "relation.jsonl"
    sample.write_text('{"text": "x"}\n')
    with pytest.raises(ValueError):
        parity.generate_parity_sample("relation", sample, tmp_path)


def test_generate_parity_sample_gnn_missing_candidates(tmp_path):
    sample = tmp_path / "gnn.jsonl"
    sample.write_text('{"query_id": "q"}\n')
    with pytest.raises(ValueError):
        parity.generate_parity_sample("gnn", sample, tmp_path)


def test_load_first_record_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError):
        parity._load_first_record(path)


def test_embedding_parity_requires_text(monkeypatch, tmp_path):
    sample = tmp_path / "embedding.jsonl"
    sample.write_text("{}\n")
    monkeypatch.setattr(parity, "load_artifact_metadata", lambda *a, **k: SimpleNamespace(version="v1", onnx_path="x", quantized_path="", config={"model_name": "stub"}))
    monkeypatch.setattr(parity.AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr(parity.ort, "InferenceSession", DummySession)
    with pytest.raises(ValueError):
        parity.generate_parity_sample("embedding", sample, tmp_path)


def test_gnn_parity_handles_empty_edges(monkeypatch, tmp_path):
    sample = tmp_path / "gnn.jsonl"
    sample.write_text(
        json.dumps(
            {
                "query_id": "q",
                "candidates": [{"node_features": [[0.1]], "edge_index": [], "edge_directions": [], "label": 0.5}],
            }
        )
        + "\n"
    )
    report = parity.generate_parity_sample("gnn", sample, tmp_path)
    assert report["outputs"]["score"] == 0.5

