"""Datasets for embedding model training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from torch.utils.data import Dataset


class EmbeddingPairDataset(Dataset):
    """Loads query/context pairs from a JSONL file for contrastive training."""

    def __init__(self, file_path: Path, limit: Optional[int] = None) -> None:
        self.file_path = file_path
        self.samples: List[Dict[str, str]] = []
        self._load(limit)

    def _load(self, limit: Optional[int]) -> None:
        with self.file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                anchor = _first_present(
                    data,
                    [
                        "anchor",
                        "query",
                        "question",
                        "text",
                    ],
                )
                positive = _first_present(data, ["positive", "response", "answer", "context"])
                if not anchor or not positive:
                    continue
                self.samples.append({"anchor": anchor, "positive": positive})
                if limit and len(self.samples) >= limit:
                    break

        if not self.samples:
            raise ValueError(f"No valid samples found in {self.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]


def build_embedding_collate_fn(tokenizer, max_length: int) -> Callable:
    """Construct a collate function that tokenizes anchor/positive pairs."""

    def _collate(batch: List[Dict[str, str]]) -> Dict[str, Dict[str, "torch.Tensor"]]:
        anchors = [sample["anchor"] for sample in batch]
        positives = [sample["positive"] for sample in batch]
        anchor_tokens = tokenizer(
            anchors,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        positive_tokens = tokenizer(
            positives,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        return {"anchor": anchor_tokens, "positive": positive_tokens}

    return _collate


def _first_present(payload: Dict[str, str], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


