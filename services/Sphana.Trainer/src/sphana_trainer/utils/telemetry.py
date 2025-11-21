"""System telemetry helpers."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Optional

try:  # pragma: no cover - psutil optional
    import psutil
except Exception:  # pragma: no cover - environments without psutil
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency detection
    import pynvml
except Exception:  # pragma: no cover - environments without NVML
    pynvml = None  # type: ignore

try:  # pragma: no cover - torch optional for metadata
    import torch
except Exception:  # pragma: no cover - tests without torch
    torch = None  # type: ignore

MB = 1024 * 1024


@dataclass
class TelemetrySample:
    """Structured telemetry data."""

    cpu_percent: float
    ram_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_allocated_mb: Optional[float] = None


class TelemetryMonitor:
    """Collects CPU/RAM/GPU utilization samples."""

    def __init__(self, device=None) -> None:
        self.device = device
        self._device_type = getattr(device, "type", None)
        self._device_index = getattr(device, "index", None)
        self._nvml = None
        self._gpu_handle = None
        self._init_nvml()

    def snapshot(self) -> Dict[str, float]:
        sample = TelemetrySample(cpu_percent=self._cpu_percent(), ram_percent=self._ram_percent())
        sample_dict = {
            "device": str(self.device) if self.device is not None else "cpu",
            "cpu_percent": sample.cpu_percent,
            "ram_percent": sample.ram_percent,
        }
        sample_dict.update(self._gpu_sample())
        return sample_dict

    def _cpu_percent(self) -> float:
        if psutil is None:
            return 0.0
        return float(psutil.cpu_percent(interval=None))

    def _ram_percent(self) -> float:
        if psutil is None:
            return 0.0
        vm = psutil.virtual_memory()
        return float(getattr(vm, "percent", 0.0))

    def _gpu_sample(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if self._device_type != "cuda":
            return stats
        stats.update(self._torch_gpu_allocation())
        if not self._nvml or self._gpu_handle is None:
            return stats
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
        except Exception:  # pragma: no cover - NVML runtime failures
            return stats
        stats["gpu_utilization"] = float(getattr(util, "gpu", 0.0))
        stats["gpu_memory_percent"] = float(getattr(util, "memory", 0.0))
        stats["gpu_memory_used_mb"] = round(float(mem.used) / MB, 2)
        stats["gpu_memory_total_mb"] = round(float(mem.total) / MB, 2)
        return stats

    def _torch_gpu_allocation(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if torch is None:
            return stats
        if not torch.cuda.is_available():  # pragma: no cover - depends on hardware
            return stats
        device_index = self._device_index
        if device_index is None:
            with contextlib.suppress(Exception):
                device_index = torch.cuda.current_device()
        if device_index is None:
            return stats
        try:
            allocated = torch.cuda.memory_allocated(device_index) / MB
        except Exception:  # pragma: no cover - guards against driver errors
            return stats
        stats["gpu_allocated_mb"] = round(float(allocated), 2)
        return stats

    def _init_nvml(self) -> None:
        if self._device_type != "cuda" or pynvml is None:
            return
        gpu_index = self._device_index
        if gpu_index is None and torch is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - only when CUDA available
                gpu_index = torch.cuda.current_device()
        if gpu_index is None:
            return
        try:
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu_handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:  # pragma: no cover - NVML init errors
            self._nvml = None
            self._gpu_handle = None


