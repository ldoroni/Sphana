"""Shared helpers for trainer runtime."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch

try:  # pragma: no cover - PyTorch version compatibility
    from torch.amp import GradScaler as AmpGradScaler
    from torch.amp import autocast as amp_autocast
    _AMP_NEEDS_DEVICE_ARG = True
except ImportError:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import GradScaler as AmpGradScaler
    from torch.cuda.amp import autocast as amp_autocast
    _AMP_NEEDS_DEVICE_ARG = False
from torch.nn.parallel import DistributedDataParallel

try:  # pragma: no cover - optional dependency guard
    import torch.distributed as dist
except OSError:  # pragma: no cover
    dist = None

PRECISION_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class TrainerContext:
    device: torch.device
    precision: Literal["fp32", "fp16", "bf16"]
    use_amp: bool
    ddp: bool
    scaler: Optional[AmpGradScaler]
    world_size: int
    global_rank: int
    managed_process_group: bool = False

    @contextmanager
    def autocast(self):  # pragma: no cover - requires specific device support
        if not (self.use_amp and self.device.type == "cuda"):
            yield
            return
        dtype = PRECISION_MAP[self.precision]
        if _AMP_NEEDS_DEVICE_ARG:
            with amp_autocast("cuda", dtype=dtype):
                yield
        else:
            with amp_autocast(dtype=dtype):
                yield

    @property
    def is_primary(self) -> bool:
        return self.global_rank == 0

    def barrier(self) -> None:  # pragma: no cover - requires torch.distributed
        if self.ddp and dist and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def wrap_model(self, model):
        if not self.ddp:
            return model
        ddp_kwargs = {}
        if self.device.type == "cuda":
            device_index = self.device.index
            if device_index is not None:
                ddp_kwargs["device_ids"] = [device_index]
                ddp_kwargs["output_device"] = device_index
        return DistributedDataParallel(model, **ddp_kwargs)

    def cleanup(self) -> None:  # pragma: no cover - requires torch.distributed
        if self.ddp and self.managed_process_group and dist and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def create_trainer_context(ddp_enabled: bool, precision: str) -> TrainerContext:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    ddp_active = ddp_enabled and world_size > 1
    managed_pg = _maybe_init_process_group(ddp_active, device)

    use_amp = precision in {"fp16", "bf16"} and device.type == "cuda"
    if use_amp and precision == "fp16":
        scaler = AmpGradScaler("cuda") if _AMP_NEEDS_DEVICE_ARG else AmpGradScaler()
    else:
        scaler = None
    return TrainerContext(
        device=device,
        precision=precision if precision in PRECISION_MAP else "fp32",
        use_amp=use_amp,
        ddp=ddp_active,
        scaler=scaler,
        world_size=world_size,
        global_rank=global_rank,
        managed_process_group=managed_pg,
    )


def save_training_state(path: Path, model, optimizer, scheduler, scaler: Optional[AmpGradScaler]) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_training_state(path: Path, model, optimizer, scheduler, scaler: Optional[AmpGradScaler]) -> None:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])


def _maybe_init_process_group(ddp_active: bool, device: torch.device) -> bool:
    if not ddp_active or not dist or not dist.is_available():  # pragma: no cover - requires torch.distributed
        return False
    if dist.is_initialized():
        return False
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    return True

