"""Unit tests for model wrappers."""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sphana_trainer.models.embedding import EmbeddingEncoder
from sphana_trainer.models.gnn import GGNNRanker


class DummyHFModel(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch, seq = input_ids.shape
        hidden = torch.ones((batch, seq, self.hidden_size), dtype=torch.float32)
        hidden += 0 if token_type_ids is None else 1
        return SimpleNamespace(last_hidden_state=hidden)

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        Path(directory, "pytorch_model.bin").write_bytes(b"ok")


def test_embedding_encoder_forward_and_save(tmp_path):
    encoder = EmbeddingEncoder(DummyHFModel())
    ids = torch.ones((2, 3), dtype=torch.long)
    mask = torch.ones_like(ids)
    token_type_ids = torch.zeros_like(ids)
    outputs = encoder(ids, mask, token_type_ids=token_type_ids)
    assert outputs.shape[0] == 2

    encoder.save_pretrained(tmp_path, pooling="cls")
    assert (tmp_path / "sphana_embedding.json").exists()

    cls_encoder = EmbeddingEncoder(DummyHFModel(), pooling="cls")
    cls_outputs = cls_encoder(ids, mask)
    assert cls_outputs.shape[0] == 2


def test_gnn_ranker_forward_and_serialization(tmp_path):
    model = GGNNRanker(input_dim=3, hidden_dim=4, num_layers=2, dropout=0.0)
    nodes = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
    edge_dirs = torch.tensor([0, 1, 2], dtype=torch.long)
    score = model(nodes, edge_index, edge_dirs)
    assert score.shape == torch.Size([1])

    empty_score = model(nodes, torch.zeros((0, 2), dtype=torch.long), torch.zeros(0, dtype=torch.long))
    assert empty_score.shape == torch.Size([1])

    none_dir_score = model(nodes, edge_index, None)
    assert none_dir_score.shape == torch.Size([1])

    checkpoint = tmp_path / "ckpt"
    saved_path = model.save(checkpoint)
    assert saved_path.exists()
    model.save_config(checkpoint)
    loaded = GGNNRanker.load(saved_path)
    assert loaded.input_dim == model.input_dim

