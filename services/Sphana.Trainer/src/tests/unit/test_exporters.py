"""Tests for ONNX exporter helpers."""

from pathlib import Path

import numpy as np
import pytest

from sphana_trainer.exporters import onnx as onnx_export
from sphana_trainer.models import GGNNRanker


def test_validate_onnx_model_invokes_session(monkeypatch, tmp_path):
    calls = {"run": 0}

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            calls["run"] += 1
            return [np.zeros((1, 4), dtype=np.float32)]

    monkeypatch.setattr(onnx_export.ort, "InferenceSession", FakeSession)
    feeds = {"input_ids": np.zeros((1, 4), dtype=np.int64)}
    reference = np.zeros((1, 4), dtype=np.float32)
    onnx_export._validate_onnx_model(tmp_path / "model.onnx", feeds, reference=reference)
    assert calls["run"] == 1


def test_validate_onnx_model_raises_on_session_failure(monkeypatch, tmp_path):
    class BrokenSession:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("invalid")

    monkeypatch.setattr(onnx_export.ort, "InferenceSession", BrokenSession)
    with pytest.raises(RuntimeError):
        onnx_export._validate_onnx_model(tmp_path / "model.onnx", {"x": np.zeros((1,))})


def test_validate_onnx_model_detects_parity(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return [np.ones((1,), dtype=np.float32)]

    monkeypatch.setattr(onnx_export.ort, "InferenceSession", FakeSession)
    feeds = {"x": np.zeros((1,), dtype=np.float32)}
    reference = np.zeros((1,), dtype=np.float32)
    with pytest.raises(AssertionError):
        onnx_export._validate_onnx_model(tmp_path / "model.onnx", feeds, reference=reference)


def test_validate_onnx_model_shape_mismatch(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return [np.zeros((2,), dtype=np.float32)]

    monkeypatch.setattr(onnx_export.ort, "InferenceSession", FakeSession)
    feeds = {"x": np.zeros((1,), dtype=np.float32)}
    reference = np.zeros((1,), dtype=np.float32)
    with pytest.raises(RuntimeError):
        onnx_export._validate_onnx_model(tmp_path / "model.onnx", feeds, reference=reference)


def test_export_gnn_ranker_validates_quantized(monkeypatch, tmp_path):
    model = GGNNRanker(input_dim=3, hidden_dim=4, num_layers=1, dropout=0.0)
    calls = {"validations": 0}

    def fake_validate(*_args, **_kwargs):
        calls["validations"] += 1

    monkeypatch.setattr(onnx_export, "_validate_onnx_model", fake_validate)
    monkeypatch.setattr(onnx_export, "_quantize_or_fallback", lambda src, dst: dst)
    onnx_path, quant_path = onnx_export.export_gnn_ranker(
        model,
        tmp_path,
        opset=17,
        max_nodes=1,
        max_edges=1,
        quantize=True,
    )
    assert calls["validations"] == 2
    assert quant_path != onnx_path

