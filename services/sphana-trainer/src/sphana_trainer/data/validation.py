"""Dataset validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from jsonschema import Draft7Validator


def validate_dataset_file(data_path: Path, schema_path: Path, limit: Optional[int] = None) -> int:
    """Validate a JSONL dataset file against a JSON schema."""

    data_path = data_path.expanduser().resolve()
    schema_path = schema_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    validator = Draft7Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    for idx, record in enumerate(_iter_jsonl(data_path)):
        if limit is not None and idx >= limit:
            break
        errors = list(validator.iter_errors(record))
        if errors:
            raise ValueError(f"Record #{idx} failed validation: {errors[0].message}")
    return idx + 1 if data_path.stat().st_size > 0 else 0


def dataset_statistics(data_path: Path, limit: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Compute simple statistics (counts, label distribution, text length)."""

    data_path = data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    total = 0
    label_counts: Dict[str, int] = {}
    length_stats = {"min": float("inf"), "max": 0.0, "sum": 0.0}

    for idx, record in enumerate(_iter_jsonl(data_path)):
        if limit is not None and idx >= limit:
            break
        total += 1
        label = (
            record.get("label")
            or record.get("predicate")
            or record.get("subject")
            or record.get("query_id")
            or "unknown"
        )
        label_counts[str(label)] = label_counts.get(str(label), 0) + 1
        text = record.get("text") or record.get("query") or ""
        length = len(text.split())
        length_stats["min"] = min(length_stats["min"], length)
        length_stats["max"] = max(length_stats["max"], length)
        length_stats["sum"] += length

    average_length = (length_stats["sum"] / total) if total else 0.0
    if length_stats["min"] == float("inf"):
        length_stats["min"] = 0.0
    return {
        "records": total,
        "labels": label_counts,
        "length": {
            "min": length_stats["min"],
            "max": length_stats["max"],
            "avg": average_length,
        },
    }


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

