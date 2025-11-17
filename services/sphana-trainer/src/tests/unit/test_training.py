"""Tests covering training error paths and helpers."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from sphana_trainer.config import EmbeddingConfig, GNNConfig, RelationExtractionConfig
from sphana_trainer.models import GGNNRanker
from sphana_trainer.training import embedding as embedding_module
from sphana_trainer.training import gnn as gnn_module
from sphana_trainer.training import relation as relation_module
from sphana_trainer.training.embedding import EmbeddingTrainer
from sphana_trainer.training.gnn import GNNTrainer, listnet_loss
from sphana_trainer.training.relation import RelationExtractionTrainer

TESTS_DATA = (Path(__file__).resolve().parents[1] / "data").resolve()


class StubTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        length = kwargs.get("max_length", 4)
        batch = len(texts)
        tensor = torch.ones((batch, length), dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": tensor.clone()}

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


class StubEmbeddingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.base_model = SimpleNamespace(
            save_pretrained=lambda directory: Path(directory).mkdir(parents=True, exist_ok=True)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        batch = input_ids.size(0)
        inputs = torch.ones((batch, 4), dtype=torch.float32)
        return self.linear(inputs)

    def to(self, device):
        return self

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


class StubRelationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, *args, **batch):
        if batch.get("input_ids") is not None:
            inputs = torch.ones((batch["input_ids"].size(0), 4), dtype=torch.float32)
        elif args:
            inputs = torch.ones((args[0].size(0), 4), dtype=torch.float32)
        else:
            inputs = torch.ones((1, 4), dtype=torch.float32)
        logits = self.linear(inputs)
        loss = logits.sum()
        return SimpleNamespace(logits=logits, loss=loss)

    def to(self, device):
        return self

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


class DummyScaler:
    def __init__(self):
        self.state = {}

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


@pytest.fixture(autouse=True)
def _disable_onnx_validation(monkeypatch):
    monkeypatch.setattr(
        "sphana_trainer.exporters.onnx._validate_onnx_model",
        lambda *args, **kwargs: None,
    )


@pytest.fixture
def patch_embedding_deps(monkeypatch):
    tokenizer = StubTokenizer()
    monkeypatch.setattr(
        "sphana_trainer.training.embedding.AutoTokenizer.from_pretrained",
        lambda *a, **k: tokenizer,
    )
    monkeypatch.setattr(
        "sphana_trainer.training.embedding.EmbeddingEncoder.from_pretrained",
        lambda *a, **k: StubEmbeddingEncoder(),
    )


@pytest.fixture
def patch_relation_deps(monkeypatch):
    tokenizer = StubTokenizer()
    monkeypatch.setattr(
        "sphana_trainer.training.relation.AutoTokenizer.from_pretrained",
        lambda *a, **k: tokenizer,
    )
    monkeypatch.setattr(
        "sphana_trainer.training.relation.AutoModelForSequenceClassification.from_pretrained",
        lambda *a, **k: StubRelationModel(),
    )


def test_embedding_trainer_requires_train_file(tmp_path, patch_embedding_deps):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = EmbeddingConfig(model_name="stub", dataset_path=str(data_dir), output_dir=str(tmp_path / "out"))
    trainer = EmbeddingTrainer(cfg, artifact_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_embedding_trainer_without_validation(tmp_path, patch_embedding_deps):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"query": "q", "positive": "p"}\n')
    resume_dir = tmp_path / "resume"
    cfg = EmbeddingConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=1,
        resume_from=str(resume_dir),
        profile_steps=1,
    )
    trainer = EmbeddingTrainer(cfg, artifact_root=tmp_path)
    state_dir = resume_dir / "checkpoint"
    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": trainer.model.state_dict(), "optimizer": None, "scheduler": None, "scaler": None},
        state_dir / "state.pt",
    )
    trainer.ctx.scaler = DummyScaler()
    result = trainer.train()
    assert "val_cosine" not in result.metrics


def test_embedding_trainer_metric_threshold_enforced(tmp_path, patch_embedding_deps, monkeypatch):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    record = '{"query": "q", "positive": "p"}\n'
    train_file.write_text(record)
    val_file.write_text(record)
    cfg = EmbeddingConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        validation_file=str(val_file),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=1,
        metric_threshold=0.9,
    )
    trainer = EmbeddingTrainer(cfg, artifact_root=tmp_path)
    monkeypatch.setattr(EmbeddingTrainer, "_evaluate", lambda self, loader: 0.1)
    with pytest.raises(RuntimeError):
        trainer.train()


def test_embedding_metric_threshold_requires_validation(tmp_path, patch_embedding_deps):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"query": "q", "positive": "p"}\n')
    cfg = EmbeddingConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=1,
        metric_threshold=0.5,
    )
    trainer = EmbeddingTrainer(cfg, artifact_root=tmp_path)
    with pytest.raises(RuntimeError):
        trainer.train()


def test_embedding_resume_skips_missing_state(tmp_path, patch_embedding_deps):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"query": "q", "positive": "p"}\n')
    cfg = EmbeddingConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=1,
        resume_from=str(tmp_path / "resume"),
    )
    trainer = EmbeddingTrainer(cfg, artifact_root=tmp_path)
    optimizer = AdamW(trainer.model.parameters(), lr=1e-4)
    trainer._maybe_resume(optimizer, scheduler=None)


def test_embedding_maybe_resume_loads_state(tmp_path, patch_embedding_deps):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"query": "q", "positive": "p"}\n')
    resume_dir = tmp_path / "resume"
    cfg = EmbeddingConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        output_dir=str(tmp_path / "out"),
        resume_from=str(resume_dir),
    )
    trainer = EmbeddingTrainer(cfg, artifact_root=tmp_path)
    state_dir = resume_dir / "checkpoint"
    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": trainer.model.state_dict(), "optimizer": None, "scheduler": None, "scaler": None},
        state_dir / "state.pt",
    )
    optimizer = AdamW(trainer.model.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1)
    trainer._maybe_resume(optimizer, scheduler)


class FakeMlflowModule:
    def __init__(self):
        self.tracking_uri = None
        self.experiment = None
        self.run_name = None
        self.tags = None
        self.logged_params = []
        self.logged_metrics = []
        self.ended = False

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, tags=None):
        self.run_name = run_name
        self.tags = tags
        return self

    def end_run(self):
        self.ended = True

    def log_params(self, params):
        self.logged_params.append(params)

    def log_metrics(self, metrics, step=None):
        self.logged_metrics.append((step, metrics))

    def log_artifact(self, path):
        pass


class DummyTelemetry:
    def __init__(self, device):
        self.device = device

    def snapshot(self):
        return {"device": str(self.device or "cpu"), "cpu_percent": 1.0, "ram_percent": 2.0}


embedding_module.TelemetryMonitor = DummyTelemetry
relation_module.TelemetryMonitor = DummyTelemetry
gnn_module.TelemetryMonitor = DummyTelemetry


def test_embedding_trainer_mlflow_logging(monkeypatch, tmp_path, patch_embedding_deps):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    record = '{"query": "q", "positive": "p"}\n'
    train_file.write_text(record)
    val_file.write_text(record)
    cfg = EmbeddingConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        validation_file=str(val_file),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=1,
        log_to_mlflow=True,
        mlflow_tracking_uri="file:///tmp/mlruns",
        mlflow_experiment="unit-test",
        mlflow_run_name="embed-test",
    )
    fake_mlflow = FakeMlflowModule()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    trainer = EmbeddingTrainer(cfg, artifact_root=tmp_path)
    trainer.train()
    assert fake_mlflow.run_name == "embed-test"
    assert fake_mlflow.tags["component"] == "embedding"
    assert fake_mlflow.logged_params
    metric_names = {name for _, metrics in fake_mlflow.logged_metrics for name in metrics.keys()}
    assert "train_loss" in metric_names
    assert "telemetry_cpu_percent" in metric_names


def test_gnn_trainer_requires_train_file(tmp_path):
    data_dir = tmp_path / "graphs"
    data_dir.mkdir()
    cfg = GNNConfig(model_name="stub", dataset_path=str(data_dir), output_dir=str(tmp_path / "out"))
    trainer = GNNTrainer(cfg, artifact_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_gnn_trainer_helpers(tmp_path):
    cfg = GNNConfig(model_name="stub", dataset_path=str(tmp_path), output_dir=str(tmp_path / "out"))
    trainer = GNNTrainer(cfg, artifact_root=tmp_path)
    trainer.ctx.device = torch.device("cpu")
    model = GGNNRanker(input_dim=3, hidden_dim=4, num_layers=1, dropout=0.0)
    candidates = [
        {"node_features": [[0.1, 0.2, 0.3]], "label": 0.5},
        {
            "node_features": [[0.2, 0.3, 0.4]],
            "edge_index": [[0, 0]],
            "edge_directions": [0],
            "label": 0.1,
        },
        {
            "node_features": [[0.3, 0.4, 0.5]],
            "edge_index": [[0, 0]],
            "label": 0.9,
        },
    ]
    scores, labels = trainer._forward_candidates(model, candidates)
    assert scores.shape[0] == len(candidates)
    zero_loss = listnet_loss(torch.empty(0, requires_grad=True), torch.empty(0), temperature=1.0)
    assert zero_loss.requires_grad


def test_gnn_trainer_train_with_scaler(tmp_path, monkeypatch):
    data_dir = TESTS_DATA / "graphs"
    resume_dir = tmp_path / "resume-gnn"
    cfg = GNNConfig(
        model_name="stub",
        dataset_path=str(data_dir),
        train_file=str(data_dir / "train.jsonl"),
        validation_file=str(data_dir / "validation.jsonl"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        resume_from=str(resume_dir),
        log_to_mlflow=True,
        profile_steps=1,
    )
    class DummyMlflowRun:
        def __init__(self):
            self.logged = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def log_params(self, params):
            self.logged.append(("params", params))

        def log_metrics(self, metrics, step=None):
            self.logged.append(("metrics", metrics, step))

    monkeypatch.setattr("sphana_trainer.training.gnn.MlflowLogger", lambda *a, **k: DummyMlflowRun())
    trainer = GNNTrainer(cfg, artifact_root=tmp_path)
    trainer.ctx.device = torch.device("cpu")
    trainer.ctx.scaler = DummyScaler()
    state_dir = resume_dir / "checkpoint"
    state_dir.mkdir(parents=True, exist_ok=True)
    dummy_model = GGNNRanker(input_dim=3, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, dropout=cfg.dropout)
    torch.save(
        {"model": dummy_model.state_dict(), "optimizer": None, "scheduler": None, "scaler": None},
        state_dir / "state.pt",
    )
    trainer.train()


def test_gnn_metric_threshold_enforced(tmp_path, monkeypatch):
    data_dir = TESTS_DATA / "graphs"
    cfg = GNNConfig(
        model_name="stub",
        dataset_path=str(data_dir),
        train_file=str(data_dir / "train.jsonl"),
        validation_file=str(data_dir / "validation.jsonl"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        metric_threshold=0.01,
    )
    trainer = GNNTrainer(cfg, artifact_root=tmp_path)
    trainer.ctx.device = torch.device("cpu")
    monkeypatch.setattr(
        "sphana_trainer.training.gnn.export_gnn_ranker",
        lambda *a, **k: (tmp_path / "gnn.onnx", tmp_path / "gnn.onnx"),
    )
    monkeypatch.setattr(GNNTrainer, "_evaluate", lambda self, model, loader: 1.0)
    with pytest.raises(RuntimeError):
        trainer.train()


def test_gnn_metric_threshold_requires_validation(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"query_id": "doc", "candidates": [{"node_features": [[0.1, 0.2]]}]}))
    cfg = GNNConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        metric_threshold=0.1,
    )
    trainer = GNNTrainer(cfg, artifact_root=tmp_path)
    with pytest.raises(RuntimeError):
        trainer.train()


def test_gnn_maybe_resume_function(tmp_path):
    cfg = GNNConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(tmp_path / "train.jsonl"),
        output_dir=str(tmp_path / "out"),
        resume_from=str(tmp_path / "resume"),
    )
    trainer = GNNTrainer(cfg, artifact_root=tmp_path)
    state_path = Path(cfg.resume_from) / "state.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    model = GGNNRanker(input_dim=3, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, dropout=cfg.dropout)
    torch.save(
        {"model": model.state_dict(), "optimizer": None, "scheduler": None, "scaler": None},
        state_path,
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)
    trainer._maybe_resume(model, optimizer)


def test_relation_trainer_requires_train_file(tmp_path, patch_relation_deps):
    data_dir = tmp_path / "relation"
    data_dir.mkdir()
    cfg = RelationExtractionConfig(model_name="stub", dataset_path=str(data_dir), output_dir=str(tmp_path / "out"))
    trainer = RelationExtractionTrainer(cfg, artifact_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_relation_trainer_early_stopping(tmp_path, patch_relation_deps, monkeypatch):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "validation.jsonl"
    record = {
        "text": "Alice built NRDB.",
        "entity1": {"text": "Alice"},
        "entity2": {"text": "NRDB"},
        "label": "founded",
    }
    for path in (train_file, val_file):
        path.write_text(json.dumps(record))

    resume_dir = tmp_path / "resume-rel"
    cfg = RelationExtractionConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        validation_file=str(val_file),
        output_dir=str(tmp_path / "out"),
        epochs=2,
        early_stopping_patience=1,
        batch_size=1,
        eval_batch_size=1,
        resume_from=str(resume_dir),
        log_to_mlflow=True,
        profile_steps=1,
    )
    dummy_path = tmp_path / "relation.onnx"
    dummy_path.write_text("onnx")
    monkeypatch.setattr(
        "sphana_trainer.training.relation.export_relation_classifier",
        lambda *a, **k: (dummy_path, dummy_path),
    )
    state_dir = resume_dir / "checkpoint"
    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": StubRelationModel().state_dict(), "optimizer": None, "scheduler": None, "scaler": None},
        state_dir / "state.pt",
    )
    class DummyMlflowRun:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def log_params(self, *_args, **_kwargs):
            pass

        def log_metrics(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("sphana_trainer.training.relation.MlflowLogger", lambda *a, **k: DummyMlflowRun())
    trainer = RelationExtractionTrainer(cfg, artifact_root=tmp_path)
    trainer.ctx.scaler = DummyScaler()
    result = trainer.train()
    assert "macro_f1" in result.metrics


def test_relation_trainer_metric_threshold_enforced(tmp_path, patch_relation_deps, monkeypatch):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    record = {
        "text": "Alice built NRDB.",
        "entity1": {"text": "Alice"},
        "entity2": {"text": "NRDB"},
        "label": "founded",
    }
    for path in (train_file, val_file):
        path.write_text(json.dumps(record))

    cfg = RelationExtractionConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        validation_file=str(val_file),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=1,
        eval_batch_size=1,
        metric_threshold=0.9,
    )
    dummy_path = tmp_path / "relation-threshold.onnx"
    dummy_path.write_text("onnx")
    monkeypatch.setattr(
        "sphana_trainer.training.relation.export_relation_classifier",
        lambda *a, **k: (dummy_path, dummy_path),
    )
    monkeypatch.setattr(
        RelationExtractionTrainer,
        "_evaluate",
        lambda self, model, loader: {"accuracy": 0.2, "macro_f1": 0.2},
    )
    trainer = RelationExtractionTrainer(cfg, artifact_root=tmp_path)
    with pytest.raises(RuntimeError):
        trainer.train()


def test_relation_metric_threshold_requires_validation(tmp_path, patch_relation_deps):
    train_file = tmp_path / "train.jsonl"
    record = {
        "text": "Alice built NRDB.",
        "entity1": {"text": "Alice"},
        "entity2": {"text": "NRDB"},
        "label": "founded",
    }
    train_file.write_text(json.dumps(record))
    cfg = RelationExtractionConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(train_file),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        metric_threshold=0.9,
    )
    trainer = RelationExtractionTrainer(cfg, artifact_root=tmp_path)
    with pytest.raises(RuntimeError):
        trainer.train()


def test_relation_maybe_resume_function(tmp_path, patch_relation_deps):
    cfg = RelationExtractionConfig(
        model_name="stub",
        dataset_path=str(tmp_path),
        train_file=str(tmp_path / "train.jsonl"),
        validation_file=str(tmp_path / "val.jsonl"),
        output_dir=str(tmp_path / "out"),
        resume_from=str(tmp_path / "resume"),
    )
    trainer = RelationExtractionTrainer(cfg, artifact_root=tmp_path)
    state_path = Path(cfg.resume_from) / "state.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Linear(4, 4)
    torch.save(
        {"model": model.state_dict(), "optimizer": None, "scheduler": None, "scaler": None},
        state_path,
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1)
    trainer._maybe_resume(model, optimizer, scheduler)

