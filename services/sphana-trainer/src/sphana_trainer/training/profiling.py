"""Helpers for optional PyTorch profiler instrumentation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from loguru import logger


class ProfilerManager:
    """Wrapper that starts/stops the PyTorch profiler when requested."""

    def __init__(self, device: torch.device, steps: int, trace_path: Path) -> None:
        self.enabled = steps > 0
        self._limit = steps
        self._profiler = None
        self._steps = 0
        self._trace_path = trace_path
        if not self.enabled:
            return
        try:
            from torch.profiler import ProfilerActivity, profile

            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available() and device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            self._profiler = profile(activities=activities, record_shapes=True)
            self._profiler.__enter__()
            logger.info("Profiler enabled for %s steps; trace -> %s", steps, trace_path)
        except Exception as exc:  # pragma: no cover - depends on torch build
            logger.warning("Unable to initialize PyTorch profiler: %s", exc)
            self.enabled = False
            self._profiler = None

    def step(self) -> None:
        if not self.enabled or not self._profiler:
            return
        self._profiler.step()
        self._steps += 1
        if self._steps >= self._limit:
            self.stop()

    def stop(self) -> None:
        if not self.enabled:
            return
        if self._profiler:
            try:
                self._profiler.export_chrome_trace(str(self._trace_path))
            except Exception as exc:  # pragma: no cover - filesystem issues
                logger.warning("Failed to export profiler trace: %s", exc)
            self._profiler.__exit__(None, None, None)
            self._profiler = None
        self.enabled = False

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        if self.enabled:
            self.stop()


