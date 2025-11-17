"""Bi-directional GGNN ranker implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn


class GGNNRanker(nn.Module):
    """Graph Neural Network ranker with GRU updates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.forward_message = nn.Linear(hidden_dim, hidden_dim)
        self.backward_message = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_directions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(node_features)
        num_nodes = h.size(0)
        if edge_directions is None:
            edge_directions = torch.zeros(edge_index.size(0), dtype=torch.long, device=h.device)

        for _ in range(self.num_layers):
            messages = torch.zeros_like(h)
            if edge_index.numel() > 0:
                src = edge_index[:, 0]
                dst = edge_index[:, 1]
                forward_mask = edge_directions == 0
                backward_mask = edge_directions == 1
                if forward_mask.any():
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst[forward_mask], h[src[forward_mask]])
                    messages = messages + self.forward_message(agg)
                if backward_mask.any():
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst[backward_mask], h[src[backward_mask]])
                    messages = messages + self.backward_message(agg)
                undirected_mask = ~(forward_mask | backward_mask)
                if undirected_mask.any():
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst[undirected_mask], h[src[undirected_mask]])
                    messages = messages + agg
            h = self.gru(messages, h)
            h = torch.relu(h)
            h = self.dropout(h)

        graph_repr, _ = torch.max(h, dim=0, keepdim=True)
        return self.readout(graph_repr).squeeze(-1)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout_rate,
            },
        }
        path = output_dir / "gnn.pt"
        torch.save(state, path)
        return path

    @classmethod
    def load(cls, checkpoint: Path) -> "GGNNRanker":
        state = torch.load(checkpoint, map_location="cpu")
        config = state["config"]
        model = cls(
            config["input_dim"],
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"],
        )
        model.load_state_dict(state["state_dict"])
        return model

    def save_config(self, output_dir: Path) -> None:
        cfg = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        }
        (output_dir / "gnn_config.json").write_text(json.dumps(cfg, indent=2))


