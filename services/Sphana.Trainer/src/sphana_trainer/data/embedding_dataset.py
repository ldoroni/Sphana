"""Datasets for embedding model training."""

from __future__ import annotations

import gzip
import json
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from torch.utils.data import Dataset
from loguru import logger


class EmbeddingPairDataset(Dataset):
    """Loads query/context pairs from a JSONL file(s) for contrastive training."""

    def __init__(self, file_pattern: Union[str, Path], limit: Optional[int] = None) -> None:
        self.file_pattern = str(file_pattern)
        self.samples: List[Dict[str, str]] = []
        self._load(limit)

    def _load(self, limit: Optional[int]) -> None:
        # Resolve files (support glob patterns)
        if any(char in self.file_pattern for char in ['*', '?', '[', ']']):
            matched_files = sorted(glob(self.file_pattern))
            if not matched_files:
                raise FileNotFoundError(f"No files matched pattern: {self.file_pattern}")
        else:
            matched_files = [self.file_pattern]
        
        # Load from all matched files
        for file_path_str in matched_files:
            file_path = Path(file_path_str)
            
            # Check if file is compressed
            if file_path_str.endswith('.gz'):
                handle = gzip.open(file_path, 'rt', encoding='utf-8')
            else:
                handle = file_path.open('r', encoding='utf-8')
            
            with handle:
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
                
                if limit and len(self.samples) >= limit:
                    break
            
            logger.debug(f"Loaded samples from {file_path_str}, total: {len(self.samples)}")

        if not self.samples:
            raise ValueError(f"No valid samples found in {self.file_pattern}")
        
        logger.info(f"Total samples loaded: {len(self.samples)} from {len(matched_files)} file(s)")

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


