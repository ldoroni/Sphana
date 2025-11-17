"""Embedding encoder wrapper with pooling + normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import torch
from torch import nn
from transformers import AutoModel


PoolingStrategy = Literal["mean", "cls"]


class EmbeddingEncoder(nn.Module):
    """Wraps a Transformer encoder with pooling + normalization."""

    def __init__(self, base_model: nn.Module, pooling: PoolingStrategy = "mean"):
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, pooling: PoolingStrategy = "mean") -> "EmbeddingEncoder":
        base_model = AutoModel.from_pretrained(model_name_or_path)
        return cls(base_model, pooling)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.base_model(**kwargs)
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        pooled = self._pool(hidden, attention_mask)
        return torch.nn.functional.normalize(pooled, p=2, dim=-1)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).type_as(hidden)
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def save_pretrained(self, output_dir: Path, pooling: Optional[PoolingStrategy] = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(output_dir)
        cfg = {"pooling": pooling or self.pooling}
        (output_dir / "sphana_embedding.json").write_text(json.dumps(cfg, indent=2))


