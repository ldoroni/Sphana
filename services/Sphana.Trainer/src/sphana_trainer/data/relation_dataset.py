"""Relation extraction dataset utilities."""

from __future__ import annotations

import gzip
import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from loguru import logger


class RelationDataset(Dataset):
    """Loads entity-marked sentences for relation classification."""

    def __init__(
        self,
        file_pattern: Union[str, Path],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int,
        allow_new_labels: bool = True,
    ) -> None:
        self.samples: List[Dict[str, torch.Tensor]] = []
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.allow_new_labels = allow_new_labels
        self.file_pattern = str(file_pattern)
        self._load()

    def _load(self) -> None:
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
                    record = json.loads(line)
                    text = record.get("text") or record.get("sentence")
                    if not text:
                        continue
                    ent1 = _entity_from_record(record, "entity1")
                    ent2 = _entity_from_record(record, "entity2")
                    label_name = record.get("label")
                    if not ent1 or not ent2 or not label_name:
                        continue
                    if label_name not in self.label2id:
                        if not self.allow_new_labels:
                            raise ValueError(f"Unknown label '{label_name}' encountered in validation set")
                        self.label2id[label_name] = len(self.label2id)
                    label_id = self.label2id[label_name]
                    marked = _insert_entity_markers(text, ent1, ent2)
                    tokenized = self.tokenizer(
                        marked,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    sample = {k: v.squeeze(0) for k, v in tokenized.items()}
                    sample["labels"] = torch.tensor(label_id, dtype=torch.long)
                    self.samples.append(sample)
            
            logger.debug(f"Loaded samples from {file_path_str}, total: {len(self.samples)}")

        if not self.samples:
            raise ValueError(f"No valid relation samples found in {self.file_pattern}")
        
        logger.info(f"Total relation samples loaded: {len(self.samples)} from {len(matched_files)} file(s)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def _entity_from_record(record: Dict, key: str) -> Optional[Dict]:
    entity = record.get(key)
    if isinstance(entity, dict):
        return {
            "text": entity.get("text") or entity.get("value"),
            "start": entity.get("start"),
            "end": entity.get("end"),
        }
    if isinstance(entity, str):
        return {"text": entity, "start": None, "end": None}
    return None


def _insert_entity_markers(text: str, ent1: Dict, ent2: Dict) -> str:
    """Wrap entity spans with markers to expose structure."""

    markers = [
        {"entity": ent1, "open": "[E1]", "close": "[/E1]"},
        {"entity": ent2, "open": "[E2]", "close": "[/E2]"},
    ]
    # Sort by start position descending to maintain offsets when inserting.
    spans = []
    for marker in markers:
        ent = marker["entity"]
        if ent.get("start") is not None and ent.get("end") is not None:
            spans.append(
                (
                    int(ent["start"]),
                    int(ent["end"]),
                    marker["open"],
                    marker["close"],
                )
            )
        else:
            text_idx = text.find(ent.get("text", ""))
            if text_idx >= 0:
                spans.append(
                    (
                        text_idx,
                        text_idx + len(ent.get("text", "")),
                        marker["open"],
                        marker["close"],
                    )
                )
    if not spans:
        return f"[E1]{ent1.get('text','')}[/E1] [E2]{ent2.get('text','')}[/E2] {text}"

    spans.sort(key=lambda item: item[0], reverse=True)
    updated = text
    for start, end, open_tok, close_tok in spans:
        updated = updated[:end] + f" {close_tok} " + updated[end:]
        updated = updated[:start] + f" {open_tok} " + updated[start:]
    return updated


