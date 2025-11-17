"""Tests for trainer context utilities."""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import amp

from sphana_trainer.training.context import (
    TrainerContext,
    create_trainer_context,
    load_training_state,
    save_training_state,
    _maybe_init_process_group,
)


def test_create_trainer_context_defaults(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    ctx = create_trainer_context(ddp_enabled=False, precision="fp32")
    assert ctx.world_size == 1
    assert not ctx.ddp
    assert not ctx.use_amp


def test_context_initializes_and_cleans_ddp(monkeypatch):
    state = {"initialized": False, "barriers": 0, "backend": None}

    def is_available():
        return True

    def is_initialized():
        return state["initialized"]

    def init_process_group(backend):
        state["initialized"] = True
        state["backend"] = backend

    def destroy_process_group():
        state["initialized"] = False

    def barrier():
        state["barriers"] += 1

    fake_dist = SimpleNamespace(
        is_available=is_available,
        is_initialized=is_initialized,
        init_process_group=init_process_group,
        destroy_process_group=destroy_process_group,
        barrier=barrier,
    )

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr("sphana_trainer.training.context.dist", fake_dist, raising=False)

    ctx = create_trainer_context(ddp_enabled=True, precision="fp32")
    assert ctx.ddp
    assert state["initialized"] is True
    expected_backend = "nccl" if torch.cuda.is_available() else "gloo"
    assert state["backend"] == expected_backend
    ctx.barrier()
    assert state["barriers"] == 1
    ctx.cleanup()
    assert state["initialized"] is False


def test_save_and_load_training_state(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    path = tmp_path / "state.pt"
    scaler = amp.GradScaler("cuda", enabled=False)
    save_training_state(path, model, optimizer, scheduler, scaler)

    new_model = torch.nn.Linear(4, 2)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    new_scheduler = torch.optim.lr_scheduler.LinearLR(new_optimizer)
    new_scaler = amp.GradScaler("cuda", enabled=False)
    load_training_state(path, new_model, new_optimizer, new_scheduler, new_scaler)


def test_load_training_state_restores_scaler(tmp_path):
    path = tmp_path / "state.pt"
    class DummyScaler:
        def __init__(self):
            self.loaded = None

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            self.loaded = state

    torch.save(
        {
            "model": torch.nn.Linear(4, 2).state_dict(),
            "optimizer": None,
            "scheduler": None,
            "scaler": {"scale": 2.0},
        },
        path,
    )
    new_model = torch.nn.Linear(4, 2)
    scaler = DummyScaler()
    load_training_state(path, new_model, None, None, scaler)
    assert scaler.loaded == {"scale": 2.0}


def test_trainer_context_wraps_model(monkeypatch):
    ctx = TrainerContext(
        device=torch.device("cuda", 0),
        precision="fp16",
        use_amp=True,
        ddp=True,
        scaler=None,
        world_size=2,
        global_rank=0,
    )
    class DummyDDP:
        def __init__(self, model, **kwargs):
            self.kwargs = kwargs
            self.model = model
    monkeypatch.setattr("sphana_trainer.training.context.DistributedDataParallel", DummyDDP)
    wrapped = ctx.wrap_model(torch.nn.Linear(4, 2))
    assert wrapped.kwargs["device_ids"] == [0]


def test_maybe_init_process_group_skips_when_initialized(monkeypatch):
    fake_dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True)
    monkeypatch.setattr("sphana_trainer.training.context.dist", fake_dist, raising=False)
    assert _maybe_init_process_group(True, torch.device("cpu")) is False

