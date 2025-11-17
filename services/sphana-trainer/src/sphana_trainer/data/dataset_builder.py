"""Utilities to derive training datasets from ingestion outputs."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional


@dataclass
class DatasetBuildResult:
    embedding_train: int
    embedding_val: int
    relation_train: int
    relation_val: int
    gnn_train: int
    gnn_val: int
    output_dir: Path


SentenceSplitter = re.compile(r"(?<=[.!?])\s+")


def build_datasets_from_ingestion(
    chunks_path: Path,
    relations_path: Path,
    output_dir: Path,
    *,
    val_ratio: float = 0.2,
    min_confidence: float = 0.0,
    seed: int = 42,
    extra_embedding: Sequence[Path] = (),
    extra_relation: Sequence[Path] = (),
    extra_gnn: Sequence[Path] = (),
    parses_dir: Optional[Path] = None,
) -> DatasetBuildResult:
    """Create embedding/relation/GNN datasets from ingestion JSONL files."""

    chunks = _load_jsonl(chunks_path)
    relations = _load_jsonl(relations_path)
    if not chunks:
        raise ValueError(f"No chunk records found at {chunks_path}")
    if not relations:
        raise ValueError(f"No relation records found at {relations_path}")

    chunk_by_id = {record["id"]: record for record in chunks if record.get("id")}
    parses = _load_parses(parses_dir) if parses_dir else {}
    embedding_records = _build_embedding_pairs(chunks)
    embedding_records.extend(_load_extra_embedding_samples(extra_embedding))
    if not embedding_records:
        raise ValueError("Embedding dataset builder produced zero samples.")

    relation_records = _build_relation_samples(relations, chunk_by_id, min_confidence, parses)
    relation_records.extend(_load_extra_relation_samples(extra_relation))
    if not relation_records:
        raise ValueError(
            "Relation dataset builder produced zero samples. "
            "Consider lowering --min-confidence or ensuring relations contain sentences."
        )

    gnn_records = _build_gnn_queries(relations, seed=seed)
    gnn_records.extend(_load_extra_gnn_samples(extra_gnn))
    if not gnn_records:
        raise ValueError("Unable to build GNN candidates from relations.")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_train, embedding_val = _write_split(
        embedding_records, output_dir / "embedding", val_ratio, seed
    )
    relation_train, relation_val = _write_split(
        relation_records, output_dir / "relation", val_ratio, seed
    )
    gnn_train, gnn_val = _write_split(gnn_records, output_dir / "gnn", val_ratio, seed)

    return DatasetBuildResult(
        embedding_train=embedding_train,
        embedding_val=embedding_val,
        relation_train=relation_train,
        relation_val=relation_val,
        gnn_train=gnn_train,
        gnn_val=gnn_val,
        output_dir=output_dir,
    )


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _build_embedding_pairs(chunks: Sequence[Dict]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for chunk in chunks:
        text = chunk.get("text") or ""
        if not text:
            continue
        sentences = [sent.strip() for sent in SentenceSplitter.split(text) if sent.strip()]
        if len(sentences) < 2:
            words = text.split()
            if len(words) < 6:
                continue
            midpoint = len(words) // 2
            sentences = [" ".join(words[:midpoint]), " ".join(words[midpoint:])]
        anchor = sentences[0]
        positive = " ".join(sentences[1:]) if len(sentences) > 1 else sentences[0]
        if anchor and positive:
            records.append({"anchor": anchor, "positive": positive})
    return records


def _build_relation_samples(
    relations: Sequence[Dict],
    chunk_by_id: Dict[str, Dict],
    min_confidence: float,
    parses: Dict[str, dict],
) -> List[Dict]:
    samples: List[Dict] = []
    for rel in relations:
        if rel.get("confidence", 0.0) < min_confidence:
            continue
        sentence = rel.get("sentence")
        if not sentence:
            chunk = chunk_by_id.get(rel.get("chunk_id", ""))
            sentence = chunk.get("text") if chunk else None
        if not sentence:
            continue
        subj = rel.get("subject")
        obj = rel.get("object")
        predicate = rel.get("predicate")
        if not (subj and obj and predicate):
            continue
        sample = {
            "text": sentence,
            "entity1": {"text": subj},
            "entity2": {"text": obj},
            "label": predicate,
        }
        parse_payload = parses.get(rel.get("chunk_id", ""))
        if parse_payload:
            selected = _select_parse_sentence(parse_payload, sentence)
            if selected:
                sample["parse"] = selected
        samples.append(sample)
    return samples


def _build_gnn_queries(relations: Sequence[Dict], *, seed: int) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for rel in relations:
        grouped.setdefault(rel.get("doc_id", "unknown"), []).append(rel)

    rng = random.Random(seed)
    records: List[Dict] = []
    for doc_id, rels in grouped.items():
        candidate = _make_graph_candidate(rels)
        if candidate is None:
            continue
        negative = _make_negative_candidate(candidate, rng)
        records.append(
            {
                "query_id": doc_id,
                "candidates": [candidate, negative],
            }
        )
    return records


def _make_graph_candidate(relations: Sequence[Dict]) -> Dict | None:
    nodes: Dict[str, int] = {}
    features: List[List[float]] = []

    def _add_node(text: str) -> int:
        if text not in nodes:
            nodes[text] = len(nodes)
            features.append(_feature_vector(text))
        return nodes[text]

    edges: List[Tuple[int, int]] = []
    confidences: List[float] = []
    for rel in relations:
        subj = rel.get("subject")
        obj = rel.get("object")
        if not (subj and obj):
            continue
        src = _add_node(subj)
        dst = _add_node(obj)
        edges.append((src, dst))
        confidences.append(float(rel.get("confidence", 0.0)))

    if not edges:
        return None
    edge_index = [
        [edge[0] for edge in edges],
        [edge[1] for edge in edges],
    ]
    directions = [0 for _ in edges]
    label = sum(confidences) / len(confidences) if confidences else 1.0
    return {
        "node_features": features,
        "edge_index": edge_index,
        "edge_directions": directions,
        "label": label,
    }


def _make_negative_candidate(candidate: Dict, rng: random.Random) -> Dict:
    neg_edges: List[Tuple[int, int]] = []
    original_edges = list(zip(candidate["edge_index"][0], candidate["edge_index"][1]))
    rng.shuffle(original_edges)
    for src, dst in original_edges:
        neg_edges.append((dst, src))
    if not neg_edges:
        return {
            "node_features": candidate["node_features"],
            "edge_index": [[], []],
            "edge_directions": [],
            "label": 0.0,
        }
    edge_index = [
        [edge[0] for edge in neg_edges],
        [edge[1] for edge in neg_edges],
    ]
    return {
        "node_features": candidate["node_features"],
        "edge_index": edge_index,
        "edge_directions": [1 for _ in neg_edges],
        "label": 0.0,
    }


def _feature_vector(text: str, dim: int = 8) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for idx in range(dim):
        byte = digest[idx]
        values.append((byte / 255.0) * 2 - 1)
    return values


def _write_split(
    records: Sequence[Dict],
    target_dir: Path,
    val_ratio: float,
    seed: int,
) -> Tuple[int, int]:
    if not records:
        raise ValueError("Cannot write split for empty dataset.")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        train, val = shuffled, []
    else:
        val_size = max(1, int(len(shuffled) * val_ratio))
        if val_size >= len(shuffled):
            val_size = len(shuffled) - 1
        val = shuffled[:val_size]
        train = shuffled[val_size:]
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target_dir / "train.jsonl", train)
    if val:
        _write_jsonl(target_dir / "validation.jsonl", val)
    else:
        (target_dir / "validation.jsonl").write_text("", encoding="utf-8")
    return len(train), len(val)


def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_parses(parses_dir: Optional[Path]) -> Dict[str, dict]:
    if not parses_dir:
        return {}
    parses: Dict[str, dict] = {}
    if not parses_dir.exists():
        return {}
    for path in parses_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "chunk_id" in data and "payload" in data:
            key = data["chunk_id"]
            value = data["payload"]
        else:
            key = path.stem
            value = data
        parses[key] = value
    return parses


def _load_extra_embedding_samples(paths: Sequence[Path]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for path in paths:
        extras = _load_jsonl(path)
        for item in extras:
            anchor = _first_present(
                item,
                [
                    "anchor",
                    "query",
                    "text",
                ],
            )
            positive = _first_present(item, ["positive", "response", "answer", "context"])
            if anchor and positive:
                records.append({"anchor": anchor, "positive": positive})
    return records


def _load_extra_relation_samples(paths: Sequence[Path]) -> List[Dict]:
    records: List[Dict] = []
    for path in paths:
        extras = _load_jsonl(path)
        for item in extras:
            text = item.get("text") or item.get("sentence")
            e1 = item.get("entity1") or item.get("subject")
            e2 = item.get("entity2") or item.get("object")
            label = item.get("label") or item.get("predicate")
            if not (text and e1 and e2 and label):
                continue
            if isinstance(e1, str):
                e1 = {"text": e1}
            if isinstance(e2, str):
                e2 = {"text": e2}
            records.append(
                {
                    "text": text,
                    "entity1": {"text": e1.get("text")},
                    "entity2": {"text": e2.get("text")},
                    "label": label,
                }
            )
    return records


def _load_extra_gnn_samples(paths: Sequence[Path]) -> List[Dict]:
    records: List[Dict] = []
    for path in paths:
        extras = _load_jsonl(path)
        for item in extras:
            if "candidates" not in item:
                continue
            records.append({"query_id": item.get("query_id") or item.get("id"), "candidates": item["candidates"]})
    return records


def _select_parse_sentence(payload: dict, sentence: str) -> Optional[dict]:
    sentences = payload.get("sentences", [])
    target = sentence.strip()
    for sent in sentences:
        if sent.get("text", "").strip() == target:
            return sent
    return sentences[0] if sentences else None


def _first_present(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


