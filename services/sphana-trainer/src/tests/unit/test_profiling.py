"""Tests for the ProfilerManager helper."""

from pathlib import Path

import torch

from sphana_trainer.training.profiling import ProfilerManager


class FakeProfiler:
    def __init__(self, trace_path: Path):
        self.trace_path = trace_path
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def step(self):
        self.steps += 1

    def export_chrome_trace(self, output):
        Path(output).write_text("trace")


def test_profiler_manager_records_trace(monkeypatch, tmp_path):
    fake = FakeProfiler(tmp_path / "trace.json")

    class DummyActivity:
        CPU = "cpu"
        CUDA = "cuda"

    monkeypatch.setattr("torch.profiler.ProfilerActivity", DummyActivity)
    monkeypatch.setattr("torch.profiler.profile", lambda *a, **k: fake)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    mgr = ProfilerManager(torch.device("cuda"), steps=2, trace_path=tmp_path / "trace.json")
    assert mgr.enabled
    mgr.step()
    mgr.step()
    assert (tmp_path / "trace.json").exists()


def test_profiler_manager_disabled(tmp_path):
    mgr = ProfilerManager(torch.device("cpu"), steps=0, trace_path=tmp_path / "noop.json")
    assert not mgr.enabled
    mgr.step()
    mgr.stop()

