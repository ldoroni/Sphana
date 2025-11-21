"""Tests for dataset loaders."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from sphana_trainer.data.embedding_dataset import (
    EmbeddingPairDataset,
    build_embedding_collate_fn,
    _first_present,
)
from sphana_trainer.data.graph_dataset import GraphQueryDataset
from sphana_trainer.data.relation_dataset import RelationDataset, _entity_from_record, _insert_entity_markers


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        length = kwargs.get("max_length", 4)
        batch = len(texts)
        tensor = torch.ones((batch, length), dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": tensor.clone()}


def test_embedding_dataset_validates_entries(tmp_path):
    data_file = tmp_path / "embed.jsonl"
    data_file.write_text(
        json.dumps({"query": "Q1", "positive": "P1"}) + "\n" + json.dumps({"text": ""})
    )
    ds = EmbeddingPairDataset(data_file)
    assert len(ds) == 1

    tokenizer = FakeTokenizer()
    collate = build_embedding_collate_fn(tokenizer, max_length=8)
    batch = collate([ds[0]])
    assert batch["anchor"]["input_ids"].shape[1] == 8


def test_embedding_dataset_raises_when_empty(tmp_path):
    data_file = tmp_path / "embed.jsonl"
    data_file.write_text("{}\n")
    with pytest.raises(ValueError):
        EmbeddingPairDataset(data_file)


def test_first_present_trims_values():
    payload = {"query": "  hello  "}
    assert _first_present(payload, ["query"]) == "hello"
    assert _first_present({}, ["missing"]) is None


def test_graph_dataset_validation(tmp_path):
    file_path = tmp_path / "graphs.jsonl"
    file_path.write_text(json.dumps({"query_id": "q1", "candidates": []}) + "\n")
    with pytest.raises(ValueError):
        GraphQueryDataset(file_path)

    file_path.write_text(
        json.dumps({"query_id": "q2", "candidates": [{"node_features": [[0.1]]}]})
    )
    dataset = GraphQueryDataset(file_path)
    assert dataset[0]["query_id"] == "q2"


def test_relation_dataset_behaviour(tmp_path):
    file_path = tmp_path / "relation.jsonl"
    bad_records = [
        {"entity1": {"text": "bad"}, "entity2": {"text": "bad"}, "label": "x"},
        {"text": "Missing label", "entity1": "A", "entity2": "B"},
    ]
    record = {
        "text": "Alice built NRDB.",
        "entity1": {"text": "Alice", "start": 0, "end": 5},
        "entity2": "NRDB",
        "label": "founder",
    }
    file_path.write_text("\n".join(json.dumps(r) for r in (*bad_records, record)))
    tokenizer = FakeTokenizer()
    label2id = {}
    dataset = RelationDataset(file_path, tokenizer, label2id, max_length=8)
    assert dataset[0]["labels"].item() == 0

    with pytest.raises(ValueError):
        RelationDataset(file_path, tokenizer, {}, max_length=8, allow_new_labels=False)

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")
    with pytest.raises(ValueError):
        RelationDataset(empty_file, tokenizer, {}, max_length=8)


def test_entity_and_marker_helpers():
    record = {"entity1": {"text": "Alice", "start": 0, "end": 5}, "entity2": "NRDB"}
    ent = _entity_from_record(record, "entity1")
    assert ent["text"] == "Alice"
    ent2 = _entity_from_record(record, "entity2")
    assert ent2["text"] == "NRDB"
    assert _entity_from_record({}, "missing") is None

    marked = _insert_entity_markers("Alice built NRDB", {"text": "X"}, {"text": "Y"})
    assert "[E1]" in marked and "[E2]" in marked

