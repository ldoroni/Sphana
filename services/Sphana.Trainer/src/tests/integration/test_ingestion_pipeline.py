"""Integration tests for ingestion pipeline."""

from pathlib import Path
import json

import pytest

from sphana_trainer.config import IngestionConfig
from sphana_trainer.data.pipeline import (
    IngestionPipeline,
    IngestionResult,
    run_ingestion,
    summarize_ingestion,
    validate_ingestion_result,
)
from sphana_trainer.tasks.ingest import IngestionTask


pytestmark = pytest.mark.ingestion
TEST_DATA = (Path(__file__).resolve().parents[1] / "data").resolve()
INGEST_SAMPLE = TEST_DATA / "ingest_sample"


def test_ingestion_pipeline_creates_cache(tmp_path):
    source = (INGEST_SAMPLE / "docs.jsonl").resolve()
    config = IngestionConfig(
        source=source,
        output_dir=tmp_path / "ingest-out",
        chunk_size=10,
        chunk_overlap=2,
        cache_enabled=True,
    )
    pipeline = IngestionPipeline(config)
    result = pipeline.run()
    assert result.chunks_path.exists()
    assert result.relations_path.exists()
    assert result.relation_count >= 0
    relations_cache = pipeline.cache_dir / "relations"
    assert relations_cache.exists()

    validate_ingestion_result(result)
    cache_mtime = result.chunks_path.stat().st_mtime
    result_cached = pipeline.run()
    assert cache_mtime == result_cached.chunks_path.stat().st_mtime
    relations_lines = result.relations_path.read_text(encoding="utf-8").strip().splitlines()
    if relations_lines:
        assert len(relations_lines) == result.relation_count


def test_ingestion_pipeline_errors_on_empty(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    config = IngestionConfig(source=empty, output_dir=tmp_path / "out", chunk_overlap=4)
    pipeline = IngestionPipeline(config)
    with pytest.raises(ValueError):
        pipeline.run()


def test_validate_ingestion_result_detects_missing(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    relations = tmp_path / "relations.jsonl"
    chunks.write_text(json.dumps({"id": "c1", "document_id": "d1", "text": "body"}) + "\n")
    relations.write_text("")
    result = IngestionResult(
        chunks_path=chunks,
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=1,
        relation_count=0,
    )
    validate_ingestion_result(result)

    missing = IngestionResult(
        chunks_path=tmp_path / "missing.jsonl",
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=0,
        chunk_count=0,
        relation_count=0,
    )
    with pytest.raises(ValueError):
        validate_ingestion_result(missing)


def test_validate_ingestion_result_schema_checks(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    relations = tmp_path / "relations.jsonl"
    good_chunk = {"id": "c1", "document_id": "d1", "text": "hello world"}
    good_relation = {
        "doc_id": "d1",
        "chunk_id": "c1",
        "subject": "Alice",
        "predicate": "founded",
        "object": "NRDB",
    }
    chunks.write_text(json.dumps(good_chunk) + "\n")
    relations.write_text(json.dumps(good_relation) + "\n")
    result = IngestionResult(
        chunks_path=chunks,
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=1,
        relation_count=1,
    )
    validate_ingestion_result(result)

    bad_chunks = tmp_path / "chunks_bad.jsonl"
    bad_chunks.write_text(json.dumps({"document_id": "d1"}) + "\n")
    bad_result = IngestionResult(
        chunks_path=bad_chunks,
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=1,
        relation_count=1,
    )
    with pytest.raises(ValueError):
        validate_ingestion_result(bad_result)

    bad_rel = tmp_path / "relations_bad.jsonl"
    bad_rel.write_text(json.dumps({"chunk_id": "c1"}) + "\n")
    bad_rel_result = IngestionResult(
        chunks_path=chunks,
        relations_path=bad_rel,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=1,
        relation_count=1,
    )
    with pytest.raises(ValueError):
        validate_ingestion_result(bad_rel_result)


def test_ingestion_pipeline_produces_chunks(tmp_path):
    input_dir = INGEST_SAMPLE.resolve()
    cfg = IngestionConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "ingest-out",
        cache_dir=tmp_path / "cache",
        chunk_size=32,
        chunk_overlap=8,
    )
    result = run_ingestion(cfg)
    assert result.document_count == 2
    assert result.chunk_count >= 2
    assert result.relation_count >= 1

    chunks_file = result.output_dir / "chunks.jsonl"
    assert chunks_file.exists()
    lines = chunks_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == result.chunk_count


def test_ingestion_cache_hits(tmp_path):
    input_dir = INGEST_SAMPLE.resolve()
    cfg = IngestionConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "ingest-out",
        cache_dir=tmp_path / "cache",
        chunk_size=32,
        chunk_overlap=8,
    )
    first = run_ingestion(cfg)
    second = run_ingestion(cfg)
    assert first.chunk_count == second.chunk_count
    assert first.relation_count == second.relation_count


def test_ingestion_task_wrapper(tmp_path):
    input_dir = INGEST_SAMPLE.resolve()
    cfg = IngestionConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "ingest-out",
        cache_dir=tmp_path / "cache",
        chunk_size=32,
        chunk_overlap=8,
    )
    task = IngestionTask(cfg)
    task.run()
    assert (cfg.output_dir / "chunks.jsonl").exists()
    assert (cfg.output_dir / "relations.jsonl").exists()


def test_summarize_ingestion_returns_label_counts(tmp_path):
    source = (INGEST_SAMPLE / "docs.jsonl").resolve()
    config = IngestionConfig(
        source=source,
        output_dir=tmp_path / "ingest-out",
        chunk_size=10,
        chunk_overlap=2,
    )
    pipeline = IngestionPipeline(config)
    result = pipeline.run(force=True)
    stats = summarize_ingestion(result)
    assert stats["documents"] == result.document_count
    assert stats["chunks"] == result.chunk_count
    assert isinstance(stats["labels"], dict)


def test_ingestion_relation_classifier(monkeypatch, tmp_path):
    source = (INGEST_SAMPLE / "docs.jsonl").resolve()
    config = IngestionConfig(
        source=source,
        output_dir=tmp_path / "ingest-out",
        chunk_size=10,
        chunk_overlap=2,
        relation_model="dummy",
    )

    class FakeClassifier:
        def __init__(self, *_args, **_kwargs):
            pass

        def classify(self, sentence):
            return "related_to", 0.9

    monkeypatch.setattr("sphana_trainer.data.pipeline.RelationClassifier", lambda *a, **k: FakeClassifier())
    pipeline = IngestionPipeline(config)
    result = pipeline.run(force=True)
    relations = [json.loads(line) for line in result.relations_path.read_text().splitlines() if line.strip()]
    assert all(rel["predicate"] == "related_to" for rel in relations)


def test_run_ingestion_with_real_wiki_source(tmp_path):
    wiki_source = (TEST_DATA / "wiki" / "docs.jsonl").resolve()
    cfg = IngestionConfig(
        source=wiki_source,
        output_dir=tmp_path / "wiki-out",
        cache_dir=tmp_path / "wiki-cache",
        chunk_size=80,
        chunk_overlap=16,
    )
    result = run_ingestion(cfg, force=True)
    assert result.document_count >= 1
    assert result.chunk_count >= result.document_count

