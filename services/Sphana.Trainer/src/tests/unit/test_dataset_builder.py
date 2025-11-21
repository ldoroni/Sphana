"""Tests for dataset builder utilities."""

import json
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sphana_trainer.cli import app
from sphana_trainer.data.dataset_builder import (
    build_datasets_from_ingestion,
    _load_jsonl,
    _build_embedding_pairs,
    _build_relation_samples,
    _make_graph_candidate,
    _make_negative_candidate,
    _write_split,
    _load_parses,
    _load_extra_relation_samples,
    _load_extra_gnn_samples,
    _first_present,
)


DATA_DIR = (Path(__file__).resolve().parents[1] / "data" / "ingestion_builder").resolve()
runner = CliRunner()


def test_build_datasets_from_ingestion(tmp_path):
    result = build_datasets_from_ingestion(
        DATA_DIR / "chunks.jsonl",
        DATA_DIR / "relations.jsonl",
        tmp_path / "derived",
        val_ratio=0.5,
        min_confidence=0.5,
        seed=123,
    )
    assert result.embedding_train >= 1
    assert (result.output_dir / "embedding" / "train.jsonl").exists()
    assert (result.output_dir / "relation" / "validation.jsonl").exists()
    assert (result.output_dir / "gnn" / "train.jsonl").exists()


def test_build_datasets_uses_parse(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    relations = tmp_path / "relations.jsonl"
    parses_dir = tmp_path / "cache" / "parses"
    parses_dir.mkdir(parents=True)
    chunk = {
        "id": "doc-chunk-0",
        "document_id": "doc",
        "text": "Alice built NRDB in a prototype environment to demo parsing.",
    }
    relation = {
        "doc_id": "doc",
        "chunk_id": "doc-chunk-0",
        "subject": "Alice",
        "object": "NRDB",
        "predicate": "built",
        "sentence": "Alice built NRDB.",
        "confidence": 0.9,
    }
    chunks.write_text(json.dumps(chunk) + "\n")
    relations.write_text(json.dumps(relation) + "\n")
    parse_payload = {"sentences": [{"text": "Alice built NRDB.", "tokens": [{"text": "Alice"}]}]}
    (parses_dir / "doc-chunk-0.json").write_text(json.dumps(parse_payload))
    result = build_datasets_from_ingestion(
        chunks,
        relations,
        tmp_path / "derived",
        parses_dir=parses_dir,
    )
    rel_train = (result.output_dir / "relation" / "train.jsonl").read_text().splitlines()
    obj = json.loads(rel_train[0])
    assert "parse" in obj


def test_build_datasets_with_extra_corpora(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    relations = tmp_path / "relations.jsonl"
    chunk = {"id": "doc-chunk-0", "document_id": "doc", "text": "Sentence one. Sentence two for positives."}
    relation = {
        "doc_id": "doc",
        "chunk_id": "doc-chunk-0",
        "subject": "Alice",
        "object": "NRDB",
        "predicate": "founder_of",
        "sentence": "Alice founded NRDB.",
        "confidence": 0.9,
    }
    chunks.write_text(json.dumps(chunk) + "\n")
    relations.write_text(json.dumps(relation) + "\n")
    extra_emb = tmp_path / "extra-emb.jsonl"
    extra_emb.write_text('{"query": "extra", "positive": "data"}\n')
    extra_rel = tmp_path / "extra-rel.jsonl"
    extra_rel.write_text(json.dumps({"text": "foo", "entity1": {"text": "a"}, "entity2": {"text": "b"}, "label": "rel"}) + "\n")
    extra_gnn = tmp_path / "extra-gnn.jsonl"
    extra_gnn.write_text(
        json.dumps(
            {"query_id": "q", "candidates": [{"node_features": [[0.1]], "edge_index": [], "edge_directions": [], "label": 0.5}]}
        )
        + "\n"
    )

    result = build_datasets_from_ingestion(
        chunks,
        relations,
        tmp_path / "derived-extra",
        extra_embedding=(extra_emb,),
        extra_relation=(extra_rel,),
        extra_gnn=(extra_gnn,),
    )
    emb_contents = (
        (result.output_dir / "embedding" / "train.jsonl").read_text()
        + (result.output_dir / "embedding" / "validation.jsonl").read_text()
    )
    rel_contents = (
        (result.output_dir / "relation" / "train.jsonl").read_text()
        + (result.output_dir / "relation" / "validation.jsonl").read_text()
    )
    gnn_contents = (
        (result.output_dir / "gnn" / "train.jsonl").read_text()
        + (result.output_dir / "gnn" / "validation.jsonl").read_text()
    )
    assert "extra" in emb_contents
    assert "foo" in rel_contents
    assert "query_id" in gnn_contents


def test_build_datasets_raises_on_missing_chunks(tmp_path):
    chunks = tmp_path / "empty.jsonl"
    relations = tmp_path / "relations.jsonl"
    chunks.write_text("")
    relations.write_text("")
    with pytest.raises(ValueError):
        build_datasets_from_ingestion(chunks, relations, tmp_path / "derived")


def test_build_datasets_raises_on_empty_relations(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    relations = tmp_path / "relations.jsonl"
    chunks.write_text(json.dumps({"id": "c", "document_id": "d", "text": "Sentence one. Sentence two."}) + "\n")
    relations.write_text("")
    with pytest.raises(ValueError):
        build_datasets_from_ingestion(chunks, relations, tmp_path / "derived")


def test_build_datasets_errors_for_short_chunks(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    relations = tmp_path / "relations.jsonl"
    chunks.write_text(json.dumps({"id": "c", "document_id": "d", "text": "short text"}) + "\n")
    relations.write_text(
        json.dumps(
            {"doc_id": "d", "chunk_id": "c", "subject": "Alice", "object": "NRDB", "predicate": "built", "sentence": "x", "confidence": 1.0}
        )
        + "\n"
    )
    with pytest.raises(ValueError):
        build_datasets_from_ingestion(chunks, relations, tmp_path / "derived")


def test_build_datasets_errors_for_missing_relations(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    relations = tmp_path / "relations.jsonl"
    chunks.write_text(json.dumps({"id": "c", "document_id": "d", "text": "Enough words for embedding sample generation here."}) + "\n")
    relations.write_text(
        json.dumps({"doc_id": "d", "chunk_id": "c", "subject": "", "object": "", "predicate": "", "confidence": 0.9}) + "\n"
    )
    with pytest.raises(ValueError):
        build_datasets_from_ingestion(chunks, relations, tmp_path / "derived")


def test_build_datasets_errors_for_empty_gnn(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    relations = tmp_path / "relations.jsonl"
    chunks.write_text(json.dumps({"id": "c", "document_id": "d", "text": "Sentence one. Sentence two for positives."}) + "\n")
    relations.write_text(
        json.dumps({"doc_id": "d", "chunk_id": "c", "subject": "", "object": "", "predicate": "", "sentence": "", "confidence": 0.9}) + "\n"
    )
    with pytest.raises(ValueError):
        build_datasets_from_ingestion(chunks, relations, tmp_path / "derived")


def test_load_jsonl_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_jsonl(tmp_path / "missing.jsonl")


def test_build_datasets_errors_for_empty_gnn_with_extra_relation(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"id": "c", "document_id": "d", "text": "Sentence one. Sentence two more words."}) + "\n")
    relations = tmp_path / "relations.jsonl"
    relations.write_text(json.dumps({"doc_id": "d", "chunk_id": "c"}) + "\n")
    extra_rel = tmp_path / "extra-rel.jsonl"
    extra_rel.write_text(json.dumps({"text": "Alice built NRDB.", "entity1": {"text": "Alice"}, "entity2": {"text": "NRDB"}, "label": "built"}) + "\n")
    with pytest.raises(ValueError):
        build_datasets_from_ingestion(
            chunks,
            relations,
            tmp_path / "derived-gnn",
            extra_relation=(extra_rel,),
        )


def test_build_embedding_pairs_skips_empty_text():
    pairs = _build_embedding_pairs([{"text": ""}])
    assert pairs == []


def test_build_relation_samples_filters_confidence():
    relations = [{"chunk_id": "c", "subject": "A", "predicate": "p", "object": "B", "confidence": 0.1}]
    samples = _build_relation_samples(relations, {}, min_confidence=0.5, parses={})
    assert samples == []


def test_make_graph_candidate_none_when_no_edges():
    assert _make_graph_candidate([{"subject": "", "object": ""}]) is None


def test_make_negative_candidate_handles_no_edges():
    candidate = {"node_features": [[0.1]], "edge_index": [[], []]}
    neg = _make_negative_candidate(candidate, random.Random(0))
    assert neg["edge_index"] == [[], []]
    assert neg["label"] == 0.0


def test_write_split_single_record(tmp_path):
    train, val = _write_split([{"x": 1}], tmp_path / "split", 0.2, seed=1)
    assert train == 1 and val == 0


def test_write_split_adjusts_val_size(tmp_path):
    records = [{"x": i} for i in range(2)]
    train, val = _write_split(records, tmp_path / "split2", val_ratio=0.9, seed=1)
    assert train == 1 and val == 1


def test_load_parses_missing_returns_empty(tmp_path):
    assert _load_parses(tmp_path / "missing") == {}


def test_load_parses_none_returns_empty():
    assert _load_parses(None) == {}


def test_load_extra_relation_samples_normalizes_strings(tmp_path):
    path = tmp_path / "extra-rel.jsonl"
    path.write_text(json.dumps({"text": "foo", "entity1": "a", "entity2": "b", "label": "rel"}) + "\n")
    samples = _load_extra_relation_samples((path,))
    assert samples[0]["entity1"]["text"] == "a"


def test_load_extra_gnn_samples_skips_invalid(tmp_path):
    path = tmp_path / "extra-gnn.jsonl"
    path.write_text("{}\n")
    assert _load_extra_gnn_samples((path,)) == []


def test_first_present_handles_missing():
    assert _first_present({}, ["a", "b"]) is None


def test_build_relation_samples_skips_missing_sentence():
    relations = [{"chunk_id": "missing", "subject": "A", "predicate": "p", "object": "B"}]
    samples = _build_relation_samples(relations, {}, min_confidence=0.0, parses={})
    assert samples == []


def test_write_split_raises_on_empty(tmp_path):
    with pytest.raises(ValueError):
        _write_split([], tmp_path / "split-empty", 0.2, seed=1)


def test_write_split_adjusts_large_val_ratio(tmp_path):
    records = [{"x": i} for i in range(2)]
    train, val = _write_split(records, tmp_path / "split3", val_ratio=1.0, seed=1)
    assert train == 1 and val == 1


def test_load_parses_reads_files(tmp_path):
    parses_dir = tmp_path / "parses"
    parses_dir.mkdir()
    (parses_dir / "chunk.json").write_text('{"chunk_id": "doc::chunk-0", "payload": {"sentences": []}}')
    parses = _load_parses(parses_dir)
    assert "doc::chunk-0" in parses


def test_load_extra_relation_samples_skips_incomplete(tmp_path):
    path = tmp_path / "extra-rel.jsonl"
    path.write_text("{}\n")
    samples = _load_extra_relation_samples((path,))
    assert samples == []


def test_dataset_build_from_ingest_cli(tmp_path):
    output_dir = tmp_path / "cli-derived"
    result = runner.invoke(
        app,
        [
            "dataset-build-from-ingest",
            str(DATA_DIR),
            "--output-dir",
            str(output_dir),
            "--min-confidence",
            "0.5",
            "--val-ratio",
            "0.5",
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "embedding" / "train.jsonl").exists()

