"""Tests for the MlflowLogger helper."""

import builtins
import sys
import types

import pytest

from sphana_trainer.utils.mlflow import MlflowLogger, _stringify_dict


class DummyMlflow:
    def __init__(self):
        self.uri = None
        self.exp = None
        self.params = None
        self.metrics = None
        self.artifacts = []
        self.started = False
        self.ended = False

    def set_tracking_uri(self, uri):
        self.uri = uri

    def set_experiment(self, name):
        self.exp = name

    def start_run(self, run_name=None, tags=None):
        self.started = True
        self.run_name = run_name
        self.tags = tags
        return object()

    def end_run(self):
        self.ended = True

    def log_params(self, params):
        self.params = params

    def log_metrics(self, metrics, step=None):
        self.metrics = (metrics, step)

    def log_artifact(self, path):
        self.artifacts.append(path)


def test_mlflow_logger_context(monkeypatch):
    dummy = DummyMlflow()
    module = types.SimpleNamespace(
        set_tracking_uri=dummy.set_tracking_uri,
        set_experiment=dummy.set_experiment,
        start_run=dummy.start_run,
        end_run=dummy.end_run,
        log_params=dummy.log_params,
        log_metrics=dummy.log_metrics,
        log_artifact=dummy.log_artifact,
    )
    monkeypatch.setitem(sys.modules, "mlflow", module)
    logger = MlflowLogger(
        True,
        tracking_uri="sqlite://",
        experiment="exp",
        run_name="run",
        tags={"stage": "test"},
    )
    with logger as active:
        assert active is logger
        logger.log_params({"a": 1, "b": object()})
        logger.log_metrics({"loss": 0.1}, step=1)
        logger.log_artifact("model.onnx")
    assert dummy.uri == "sqlite://"
    assert dummy.exp == "exp"
    assert dummy.params["a"] == 1
    assert isinstance(dummy.params["b"], str)
    assert dummy.metrics == ({"loss": 0.1}, 1)
    assert dummy.artifacts == ["model.onnx"]
    assert dummy.ended


def test_mlflow_logger_disabled():
    logger = MlflowLogger(False, tracking_uri=None, experiment=None, run_name=None)
    with logger as active:
        assert active is None


def test_mlflow_logger_missing_dependency(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("mlflow", None)
    logger = MlflowLogger(True, tracking_uri=None, experiment=None, run_name=None)
    with pytest.raises(RuntimeError):
        with logger:
            pass


def test_stringify_dict():
    payload = {"a": 1, "b": object()}
    formatted = _stringify_dict(payload)
    assert formatted["a"] == 1
    assert isinstance(formatted["b"], str)

