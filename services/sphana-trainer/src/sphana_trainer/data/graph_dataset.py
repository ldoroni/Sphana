"""Graph dataset utilities for the GNN ranker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset


class GraphQueryDataset(Dataset):
    """Loads listwise query candidates for GNN ranking."""

    def __init__(self, file_path: Path) -> None:
        self.items: List[Dict] = []
        self._load(file_path)

    def _load(self, file_path: Path) -> None:
        with file_path.open("r", encoding="utf-8") as handle:
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

        if not self.items:
            raise ValueError(f"No GNN query candidates found in {file_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


