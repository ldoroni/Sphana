"""Graph dataset utilities for the GNN ranker."""

from __future__ import annotations

import gzip
import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Union

from torch.utils.data import Dataset
from loguru import logger


class GraphQueryDataset(Dataset):
    """Loads listwise query candidates for GNN ranking."""

    def __init__(self, file_pattern: Union[str, Path]) -> None:
        self.items: List[Dict] = []
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
                    payload = json.loads(line)
                    candidates = payload.get("candidates") or []
                    if not candidates:
                        continue
                    self.items.append(
                        {
                            "query_id": payload.get("query_id"),
                            "candidates": candidates,
                        }
                    )
            
            logger.debug(f"Loaded samples from {file_path_str}, total: {len(self.items)}")

        if not self.items:
            raise ValueError(f"No GNN query candidates found in {self.file_pattern}")
        
        logger.info(f"Total GNN samples loaded: {len(self.items)} from {len(matched_files)} file(s)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


