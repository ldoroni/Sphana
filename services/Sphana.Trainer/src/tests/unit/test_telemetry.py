from types import SimpleNamespace

import pytest

from sphana_trainer.utils import telemetry as telemetry_mod
from sphana_trainer.utils.telemetry import TelemetryMonitor


def test_telemetry_snapshot_cpu_only(monkeypatch):
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 12.5,
        virtual_memory=lambda: SimpleNamespace(percent=45.0),
    )
    monkeypatch.setattr(telemetry_mod, "psutil", fake_psutil, raising=False)
    monitor = TelemetryMonitor(SimpleNamespace(type="cpu", index=None))
    stats = monitor.snapshot()
    assert stats["cpu_percent"] == 12.5
    assert stats["ram_percent"] == 45.0
    assert "gpu_utilization" not in stats


def test_telemetry_snapshot_gpu_with_nvml(monkeypatch):
    class FakeNvml:
        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            return index

        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle):
            return SimpleNamespace(gpu=70, memory=55)

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            return SimpleNamespace(used=128 * telemetry_mod.MB, total=512 * telemetry_mod.MB)

    monkeypatch.setattr(telemetry_mod, "pynvml", FakeNvml, raising=False)
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 10.0,
        virtual_memory=lambda: SimpleNamespace(percent=40.0),
    )
    monkeypatch.setattr(telemetry_mod, "psutil", fake_psutil, raising=False)
    monitor = TelemetryMonitor(SimpleNamespace(type="cuda", index=0))
    stats = monitor.snapshot()
    assert stats["gpu_utilization"] == 70.0
    assert stats["gpu_memory_used_mb"] == 128.0
    assert stats["gpu_memory_total_mb"] == 512.0


def test_telemetry_handles_missing_nvml(monkeypatch):
    monkeypatch.setattr(telemetry_mod, "pynvml", None, raising=False)
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 5.0,
        virtual_memory=lambda: SimpleNamespace(percent=30.0),
    )
    monkeypatch.setattr(telemetry_mod, "psutil", fake_psutil, raising=False)
    monitor = TelemetryMonitor(SimpleNamespace(type="cuda", index=0))
    stats = monitor.snapshot()
    assert "gpu_utilization" not in stats
    assert stats["cpu_percent"] == 5.0


def test_telemetry_without_torch(monkeypatch):
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 8.0,
        virtual_memory=lambda: SimpleNamespace(percent=20.0),
    )
    monkeypatch.setattr(telemetry_mod, "psutil", fake_psutil, raising=False)
    monkeypatch.setattr(telemetry_mod, "torch", None, raising=False)
    monkeypatch.setattr(telemetry_mod, "pynvml", None, raising=False)
    stats = TelemetryMonitor(SimpleNamespace(type="cuda", index=None)).snapshot()
    assert stats["cpu_percent"] == 8.0
    assert "gpu_allocated_mb" not in stats


def test_telemetry_missing_device_index(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return None

    fake_torch = SimpleNamespace(cuda=FakeCuda)
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 9.0,
        virtual_memory=lambda: SimpleNamespace(percent=25.0),
    )
    monkeypatch.setattr(telemetry_mod, "torch", fake_torch, raising=False)
    monkeypatch.setattr(telemetry_mod, "psutil", fake_psutil, raising=False)
    monkeypatch.setattr(telemetry_mod, "pynvml", None, raising=False)
    stats = TelemetryMonitor(SimpleNamespace(type="cuda", index=None)).snapshot()
    assert stats["cpu_percent"] == 9.0
    assert "gpu_allocated_mb" not in stats


def test_telemetry_init_nvml_without_index(monkeypatch):
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 6.0,
        virtual_memory=lambda: SimpleNamespace(percent=15.0),
    )
    monkeypatch.setattr(telemetry_mod, "psutil", fake_psutil, raising=False)
    fake_nvml = SimpleNamespace()
    monkeypatch.setattr(telemetry_mod, "pynvml", fake_nvml, raising=False)
    monkeypatch.setattr(telemetry_mod, "torch", None, raising=False)
    stats = TelemetryMonitor(SimpleNamespace(type="cuda", index=None)).snapshot()
    assert stats["cpu_percent"] == 6.0


def test_telemetry_without_psutil(monkeypatch):
    monkeypatch.setattr(telemetry_mod, "psutil", None, raising=False)
    stats = TelemetryMonitor(SimpleNamespace(type="cpu", index=None)).snapshot()
    assert stats["cpu_percent"] == 0.0
    assert stats["ram_percent"] == 0.0

