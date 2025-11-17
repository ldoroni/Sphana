"""Unit tests for ingestion pipeline helpers."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sphana_trainer.config import IngestionConfig
from sphana_trainer.data import pipeline
from sphana_trainer.data.pipeline import (
    IngestionResult,
    _apply_relation_classifier,
    _chunk_text,
    _iter_jsonl,
    _load_cached_chunks,
    _load_cached_relations,
    _load_jsonl,
    _load_source_documents,
    _scan_documents,
    _store_cached_chunks,
    _store_cached_relations,
    run_ingestion,
    validate_ingestion_result,
    validate_with_schema,
)


def test_scan_documents_filters(tmp_path):
    (tmp_path / "doc.txt").write_text("Hello world")
    (tmp_path / "skip.csv").write_text("nope")
    (tmp_path / "empty.txt").write_text("   ")
    sub = tmp_path / "dir"
    sub.mkdir()
    (sub / "nested.md").write_text("Nested text")

    docs = list(_scan_documents(tmp_path))
    ids = [doc for doc, _ in docs]
    assert "doc" in ids and "nested" in ids
    assert "skip" not in ids


def test_chunk_text_handles_edge_cases():
    assert _chunk_text("", 32, 8) == []
    text = " ".join(str(i) for i in range(10))
    chunks = _chunk_text(text, chunk_size=4, overlap=2)
    assert chunks[0].startswith("0 1 2 3")


def test_cache_helpers(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_key = "abc"
    assert _load_cached_chunks(cache_dir, cache_key) is None
    sample = [{"doc_id": "a", "chunk_id": "a-1", "text": "hi"}]
    _store_cached_chunks(cache_dir, cache_key, sample)
    loaded = _load_cached_chunks(cache_dir, cache_key)
    assert loaded == sample


def test_regex_relation_extractor_matches_entities():
    cfg = IngestionConfig(parser="simple", relation_threshold=0.9)
    extractor = pipeline._create_relation_extractor(cfg)
    relations, _ = extractor.extract("doc", "doc::chunk-0", "Alice founded NRDB before Bob succeeded NRDB")
    assert relations
    assert relations[0]["subject"] == "Alice"
    assert relations[0]["object"] == "NRDB"


def test_spacy_parser_requires_dependency(monkeypatch):
    cfg = IngestionConfig(parser="spacy")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("spaCy missing")

    monkeypatch.setattr(pipeline, "_load_spacy_model", _raise)
    with pytest.raises(RuntimeError):
        pipeline._create_relation_extractor(cfg)


def test_apply_relation_classifier_filters_low_confidence():
    relations = [
        {
            "doc_id": "d1",
            "chunk_id": "d1::0",
            "subject": "Alice",
            "predicate": "founded",
            "object": "NRDB",
            "sentence": "Alice founded NRDB",
        }
    ]

    class DummyClassifier:
        def classify(self, sentence):
            return "founded", 0.1

    filtered = pipeline._apply_relation_classifier(relations, DummyClassifier(), threshold=0.5)
    assert filtered == []


def test_parse_cache_roundtrip(tmp_path):
    payload = {"sentences": [{"text": "Hello world", "tokens": [{"text": "Hello"}]}]}
    pipeline._store_cached_parse(tmp_path, "chunk-1", payload)
    loaded = pipeline.load_cached_parse(tmp_path, "chunk-1")
    assert loaded == payload


def test_run_ingestion_requires_input_dir(tmp_path):
    cfg = IngestionConfig(source=None, input_dir=None, output_dir=tmp_path / "out")
    with pytest.raises(ValueError):
        pipeline.run_ingestion(cfg)


def test_run_ingestion_stores_parse(monkeypatch, tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text('{"id": "doc", "text": "Alice built NRDB."}\n')
    cfg = IngestionConfig(
        source=source,
        output_dir=tmp_path / "out",
        parser="spacy",
        parser_model="dummy",
        relation_threshold=0.0,
        cache_enabled=False,
    )

    class FakeExtractor:
        def extract(self, doc_id, chunk_id, text):
            rel = {"doc_id": doc_id, "chunk_id": chunk_id, "subject": "Alice", "object": "NRDB", "predicate": "built", "sentence": text}
            return [rel], {"sentences": [{"text": text}]}

    monkeypatch.setattr(pipeline, "_create_relation_extractor", lambda cfg: FakeExtractor())
    result = pipeline.run_ingestion(cfg)
    parse_dir = (cfg.output_dir / "cache" / "parses")
    assert any(parse_dir.glob("*.json"))
    assert result.document_count == 1


def test_validate_ingestion_result_missing_relations(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"id": "c1", "document_id": "d1", "text": "body"}\n')
    result = IngestionResult(
        chunks_path=chunks,
        relations_path=tmp_path / "missing.jsonl",
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=1,
        relation_count=0,
    )
    with pytest.raises(ValueError):
        pipeline.validate_ingestion_result(result)


def test_iter_jsonl_missing(tmp_path):
    assert list(_iter_jsonl(tmp_path / "missing.jsonl")) == []


def test_iter_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"foo": "bar"}\n\n')
    rows = list(_iter_jsonl(path))
    assert len(rows) == 1


def test_load_jsonl_raises_for_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError):
        _load_jsonl(path)


def test_spacy_extractor_initializes(monkeypatch):
    cfg = IngestionConfig(parser="spacy", parser_model="model")
    monkeypatch.setattr(pipeline, "_load_spacy_model", lambda name: lambda text: SimpleNamespace(sents=[]))
    extractor = pipeline._create_relation_extractor(cfg)
    assert isinstance(extractor, pipeline._SpacyRelationExtractor)


def test_stanza_extractor_initializes(monkeypatch):
    cfg = IngestionConfig(parser="stanza", language="en")
    monkeypatch.setattr(pipeline, "_build_stanza_pipeline", lambda lang: lambda text: SimpleNamespace(sentences=[]))
    extractor = pipeline._create_relation_extractor(cfg)
    assert isinstance(extractor, pipeline._StanzaRelationExtractor)


def test_ingestion_pipeline_loads_calibration(monkeypatch, tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text('{"rel": {"scale": 1, "bias": 0}}')
    cfg = IngestionConfig(
        source=tmp_path / "docs.jsonl",
        output_dir=tmp_path / "out",
        relation_calibration=str(cal),
        relation_model="stub",
    )
    monkeypatch.setattr(pipeline.RelationClassifier, "__init__", lambda self, *a, **k: None)
    pipe = pipeline.IngestionPipeline(cfg)
    assert pipe.classifier is not None


def test_ingestion_pipeline_raises_when_no_chunks(monkeypatch, tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text('{"id": "1", "text": ""}\n')
    cfg = IngestionConfig(source=source, output_dir=tmp_path / "out")
    pipe = pipeline.IngestionPipeline(cfg)
    with pytest.raises(ValueError):
        pipe.run(force=True)


def test_apply_relation_classifier_without_classifier():
    relations = [{"predicate": "related_to"}]
    updated = _apply_relation_classifier(relations, None, threshold=0.7)
    assert updated[0]["confidence"] == 0.7


def test_run_ingestion_requires_source_or_input(tmp_path):
    cfg = IngestionConfig(output_dir=tmp_path / "out")
    with pytest.raises(ValueError):
        run_ingestion(cfg)


def test_validate_ingestion_missing_files(tmp_path):
    result = IngestionResult(
        chunks_path=tmp_path / "chunks.jsonl",
        relations_path=tmp_path / "relations.jsonl",
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=1,
        relation_count=0,
    )
    with pytest.raises(ValueError):
        validate_ingestion_result(result)


def test_validate_ingestion_zero_chunks(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("")
    relations = tmp_path / "relations.jsonl"
    relations.write_text("")
    result = IngestionResult(
        chunks_path=chunks,
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=0,
        relation_count=0,
    )
    with pytest.raises(ValueError):
        validate_ingestion_result(result)


def test_validate_ingestion_count_mismatch(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"id": "c", "document_id": "d", "text": "body"}\n')
    relations = tmp_path / "relations.jsonl"
    relations.write_text("")
    result = IngestionResult(
        chunks_path=chunks,
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=2,
        relation_count=0,
    )
    with pytest.raises(ValueError):
        validate_ingestion_result(result)


def test_validate_with_schema_errors(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"id": "c"}\n')
    relations = tmp_path / "relations.jsonl"
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
    schema = tmp_path / "chunk.schema.json"
    schema.write_text('{"type": "object", "required": ["id", "text"]}')
    with pytest.raises(ValueError):
        validate_with_schema(result, schema, None)


def test_load_source_documents_from_json(tmp_path):
    data = {"documents": [{"id": "Doc 1", "text": " Hello "}, "Fallback text"]}
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(data))
    docs = _load_source_documents(path)
    assert len(docs) == 2
    assert docs[0][0] == "doc-1"


def test_load_source_documents_raises_when_empty(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text("")
    with pytest.raises(ValueError):
        _load_source_documents(path)


def test_cached_relations_roundtrip(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = "abc"
    _store_cached_relations(cache_dir, key, [{"chunk_id": "c"}])
    loaded = _load_cached_relations(cache_dir, key)
    assert loaded == [{"chunk_id": "c"}]


def test_validate_ingestion_zero_documents(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"id": "c", "document_id": "d", "text": "body"}\n')
    relations = tmp_path / "relations.jsonl"
    relations.write_text("")
    result = IngestionResult(
        chunks_path=chunks,
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=0,
        chunk_count=1,
        relation_count=0,
    )
    with pytest.raises(ValueError):
        validate_ingestion_result(result)


def test_validate_ingestion_relation_mismatch(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"id": "c", "document_id": "d", "text": "body"}\n')
    relations = tmp_path / "relations.jsonl"
    relations.write_text('{"doc_id": "d", "chunk_id": "c", "subject": "a", "predicate": "p", "object": "b"}\n')
    result = IngestionResult(
        chunks_path=chunks,
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=1,
        relation_count=2,
    )
    with pytest.raises(ValueError):
        validate_ingestion_result(result)


def test_validate_with_schema_relation_errors(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"id": "c", "document_id": "d", "text": "body"}\n')
    relations = tmp_path / "relations.jsonl"
    relations.write_text('{"doc_id": "d"}\n')
    result = IngestionResult(
        chunks_path=chunks,
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=1,
        relation_count=0,
    )
    schema = tmp_path / "rel.schema.json"
    schema.write_text('{"type": "object", "required": ["doc_id", "chunk_id"]}')
    with pytest.raises(ValueError):
        validate_with_schema(result, None, schema)


def test_load_source_documents_handles_non_list(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps({"text": "Hello"}))
    docs = _load_source_documents(path)
    assert docs


def test_load_source_documents_skips_missing_text(tmp_path):
    path = tmp_path / "docs2.json"
    path.write_text(json.dumps([{"id": "doc-1"}, {"text": "hi"}]))
    docs = _load_source_documents(path)
    assert len(docs) == 1


def test_store_cached_parse_skips_empty(tmp_path):
    cache = tmp_path / "cache"
    pipeline._store_cached_parse(cache, "chunk-1", {})
    assert not (cache / "chunk-1.json").exists()


def test_store_cached_parse_roundtrip(tmp_path):
    cache = tmp_path / "cache"
    payload = {"sentences": [{"text": "hello"}]}
    pipeline._store_cached_parse(cache, "chunk-1", payload)
    assert pipeline.load_cached_parse(cache, "chunk-1") == payload


def test_load_cached_parse_missing(tmp_path):
    assert pipeline.load_cached_parse(tmp_path / "cache", "missing") is None


def test_load_cached_parse_legacy_format(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    name = hashlib.md5("legacy".encode("utf-8")).hexdigest()
    (cache / f"{name}.json").write_text('{"sentences": []}')
    result = pipeline.load_cached_parse(cache, "legacy")
    assert result == {"sentences": []}


def test_load_source_documents_wraps_scalar(tmp_path):
    path = tmp_path / "scalar.json"
    path.write_text(json.dumps("plain text"))
    docs = _load_source_documents(path)
    assert docs[0][0].startswith("doc-")


def test_ingestion_pipeline_uses_cached_relations(tmp_path, monkeypatch):
    source = tmp_path / "docs.jsonl"
    source.write_text('{"id": "doc1", "text": "Alice built NRDB."}\n')
    cfg = IngestionConfig(source=source, output_dir=tmp_path / "ingest")
    pipe = pipeline.IngestionPipeline(cfg)
    pipe.run(force=True)
    (pipe.cache_dir / "chunks.jsonl").unlink()

    def _fail(*_args, **_kwargs):
        raise RuntimeError("should not run extractor")

    pipe.extractor.extract = _fail
    pipe.run(force=False)


def test_relation_classifier_applies_calibration(monkeypatch):
    class DummyTokenizer:
        def __call__(self, texts, **kwargs):
            length = kwargs.get("max_length", 4)
            tensor = torch.zeros((1, length), dtype=torch.long)
            return {"input_ids": tensor, "attention_mask": tensor.clone()}

    class DummyModel:
        config = SimpleNamespace(id2label={0: "neg", 1: "pos"})

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **batch):
            return SimpleNamespace(logits=torch.tensor([[0.2, 0.8]]))

    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr(
        pipeline.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **k: DummyModel(),
    )
    clf = pipeline.RelationClassifier("model", max_length=4, calibration={"pos": {"scale": 1.0, "bias": 0.0}})
    label, score = clf.classify("text")
    assert label in {"pos", "1"}
    assert 0.0 <= score <= 1.0


def test_ingestion_pipeline_requires_source(tmp_path):
    cfg = IngestionConfig(output_dir=tmp_path / "ingest")
    pipe = pipeline.IngestionPipeline(cfg)
    with pytest.raises(ValueError):
        pipe.run()


def test_ingestion_pipeline_stores_parse_payload(monkeypatch, tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text('{"id": "doc1", "text": "Sample text"}\n')
    cfg = IngestionConfig(source=source, output_dir=tmp_path / "parse-out", parser="spacy")

    class DummyExtractor:
        def extract(self, doc_id, chunk_id, text):
            return [], {"sentences": [{"text": text}]}

    monkeypatch.setattr("sphana_trainer.data.pipeline._create_relation_extractor", lambda cfg: DummyExtractor())
    pipe = pipeline.IngestionPipeline(cfg)
    pipe.run(force=True)
    parses = list((pipe.cache_dir / "parses").glob("*.json"))
    assert parses