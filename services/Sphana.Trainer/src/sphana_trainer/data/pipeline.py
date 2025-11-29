"""Document ingestion and preprocessing pipeline."""

from __future__ import annotations

import hashlib
import math
import json
import re
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from jsonschema import Draft7Validator
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sphana_trainer.config import IngestionConfig
from sphana_trainer.utils.progress import ProgressTracker, _format_duration


def _load_jsonl(path: Path) -> List[dict]:
    docs: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    if not docs:
        raise ValueError(f"No documents found in {path}")
    return docs


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


class _RegexRelationExtractor:
    """Simple regex-based relation extractor used for smoke testing."""

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.pattern = re.compile(r"(?P<subj>[A-Z][\w]+)\s+(?P<pred>[\w\s]+?)\s+(?P<obj>[A-Z][\w]+)")

    def extract(self, doc_id: str, chunk_id: str, text: str) -> Tuple[List[dict], Optional[dict]]:
        records: List[dict] = []
        for match in self.pattern.finditer(text):
            predicate = match.group("pred").strip()[:64] or "related_to"
            records.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "subject": match.group("subj"),
                    "predicate": predicate,
                    "object": match.group("obj"),
                    "confidence": self.threshold,
                    "sentence": text,
                }
            )
        return records, None


class _SpacyRelationExtractor:
    """Dependency-based extractor powered by spaCy."""

    def __init__(self, model_name: str, threshold: float) -> None:
        self._nlp = _load_spacy_model(model_name)
        self.threshold = float(threshold)
        self.last_parse: Optional[dict] = None

    def extract(self, doc_id: str, chunk_id: str, text: str) -> Tuple[List[dict], Optional[dict]]:  # pragma: no cover - requires spaCy
        doc = self._nlp(text)
        records: List[dict] = []
        sentences_meta = []
        for sent in doc.sents:
            subj = None
            obj = None
            predicate = (sent.root.lemma_ if sent.root else "related_to") or "related_to"
            for token in sent:
                if token.dep_ in {"nsubj", "nsubjpass"}:
                    subj = token.text
                elif token.dep_ in {"dobj", "pobj", "attr", "obl", "oprd"}:
                    obj = token.text
            if subj and obj:
                sentences_meta.append(
                    {
                        "text": sent.text.strip(),
                        "tokens": [
                            {
                                "text": token.text,
                                "lemma": token.lemma_,
                                "dep": token.dep_,
                                "pos": token.pos_,
                                "head_offset": (token.head.i - sent.start) if token.head else 0,
                            }
                            for token in sent
                        ],
                    }
                )
                records.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "subject": subj,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": self.threshold,
                        "sentence": sent.text.strip(),
                    }
                )
        self.last_parse = {"sentences": sentences_meta}
        return records, self.last_parse


class _StanzaRelationExtractor:
    """Dependency-based extractor powered by Stanza."""

    def __init__(self, language: str, threshold: float) -> None:
        self._pipeline = _build_stanza_pipeline(language)
        self.threshold = float(threshold)
        self.last_parse: Optional[dict] = None

    def extract(self, doc_id: str, chunk_id: str, text: str) -> Tuple[List[dict], Optional[dict]]:  # pragma: no cover - requires Stanza
        doc = self._pipeline(text)
        records: List[dict] = []
        sentences_meta = []
        for sentence in doc.sentences:
            subj = None
            obj = None
            predicate = None
            tokens_meta = []
            for word in sentence.words:
                tokens_meta.append(
                    {
                        "text": word.text,
                        "lemma": word.lemma,
                        "deprel": word.deprel,
                        "pos": word.xpos,
                        "head": word.head,
                    }
                )
                if word.deprel in {"nsubj", "nsubj:pass"}:
                    subj = word.text
                    head_idx = word.head - 1
                    if 0 <= head_idx < len(sentence.words):
                        predicate = sentence.words[head_idx].lemma or sentence.words[head_idx].text
                elif word.deprel in {"obj", "obl", "iobj"}:
                    obj = word.text
            if subj and obj:
                sentences_meta.append(
                    {
                        "text": " ".join(w.text for w in sentence.words),
                        "tokens": tokens_meta,
                    }
                )
                records.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "subject": subj,
                        "predicate": predicate or "related_to",
                        "object": obj,
                        "confidence": self.threshold,
                        "sentence": " ".join(w.text for w in sentence.words),
                    }
                )
        self.last_parse = {"sentences": sentences_meta}
        return records, self.last_parse


class RelationClassifier:  # pragma: no cover - requires heavyweight HF models
    """Optional Hugging Face classifier applied to extracted relations."""

    def __init__(self, model_name: str, max_length: int, calibration: Optional[dict] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Automatic device detection (same as training commands)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            logger.info(f"RelationClassifier using GPU: {self.device}")
        else:
            logger.info("RelationClassifier using CPU")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.id2label = self.model.config.id2label or {}
        self.max_length = max_length
        self.calibration = calibration or {}

    def classify(self, sentence: str) -> Tuple[str, float]:
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            score, idx = torch.max(probs, dim=-1)
        label_idx = idx.item()
        label = self.id2label.get(label_idx, str(label_idx))
        score_value = float(score.item())
        params = self.calibration.get(label)
        if params:
            scale = params.get("scale", 1.0)
            bias = params.get("bias", 0.0)
            score_value = 1.0 / (1.0 + math.exp(-(score_value * scale + bias)))
        return label, score_value


def _create_relation_extractor(config: IngestionConfig):
    if config.parser == "simple":
        return _RegexRelationExtractor(config.relation_threshold)
    if config.parser == "spacy":
        return _SpacyRelationExtractor(config.parser_model, config.relation_threshold)
    if config.parser == "stanza":
        return _StanzaRelationExtractor(config.language, config.relation_threshold)
    raise ValueError(f"Unsupported parser backend '{config.parser}'")  # pragma: no cover - parser choices validated


def _load_spacy_model(model_name: str):  # pragma: no cover - optional dependency
    try:
        import spacy  # type: ignore
    except ImportError as exc:
        raise RuntimeError("spaCy is required for parser='spacy'. Install it via `pip install spacy`.") from exc
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with `python -m spacy download {model_name}`."
        ) from exc


def _build_stanza_pipeline(language: str):  # pragma: no cover - optional dependency
    """Build Stanza pipeline with automatic GPU detection."""
    try:
        import stanza  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Stanza is required for parser='stanza'. Install it via `pip install stanza`.") from exc
    
    # Automatic device detection (same as training commands)
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        logger.info("Stanza parser using GPU")
    else:
        logger.info("Stanza parser using CPU")
    
    try:
        return stanza.Pipeline(
            lang=language,
            processors="tokenize,pos,lemma,depparse",
            tokenize_no_ssplit=False,
            use_gpu=use_gpu,
            verbose=False,
        )
    except stanza.resources.common.ResourceNotFoundError as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            f"Stanza resources for '{language}' not found. "
            f"Download them with `python -m stanza.download {language}`."
        ) from exc


def _apply_relation_classifier(
    relations: List[dict],
    classifier: Optional[RelationClassifier],
    threshold: float,
) -> List[dict]:
    if classifier is None:
        for rel in relations:
            rel.setdefault("confidence", threshold)
        return relations

    filtered: List[dict] = []
    for rel in relations:
        sentence = rel.get("sentence") or rel.get("text") or ""
        label, score = classifier.classify(sentence)
        if score < threshold:
            continue
        rel["predicate"] = label
        rel["confidence"] = score
        filtered.append(rel)
    return filtered


@dataclass
class IngestionResult:
    chunks_path: Path
    relations_path: Path
    cache_dir: Path
    output_dir: Path
    document_count: int = 0
    chunk_count: int = 0
    relation_count: int = 0


def _cache_key(config: IngestionConfig) -> str:
    source = config.source or config.input_dir or Path("unknown")
    payload = (
        f"{source.resolve()}:{config.chunk_size}:{config.chunk_overlap}:"
        f"{config.parser}:{config.parser_model}:{config.language}:{config.relation_threshold}"
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


class IngestionPipeline:
    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir.expanduser().resolve()
        self.cache_dir = self.output_dir / "cache" / _cache_key(config)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = _create_relation_extractor(config)
        calibration = None
        if config.relation_calibration:
            calibration_path = config.relation_calibration
            calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        self.classifier = (
            RelationClassifier(
                config.relation_model,
                config.relation_max_length,
                calibration=calibration
            )
            if config.relation_model
            else None
        )

    def run(self, force: bool = False) -> IngestionResult:
        start = perf_counter()
        if self.config.cache_enabled and self.cache_dir.exists() and not force:
            chunks_path = self.cache_dir / "chunks.jsonl"
            relations_path = self.cache_dir / "relations.jsonl"
            if chunks_path.exists() and relations_path.exists():
                return IngestionResult(
                    chunks_path=chunks_path,
                    relations_path=relations_path,
                    cache_dir=self.cache_dir,
                    output_dir=self.output_dir,
                )

        if self.config.source:
            docs = _load_jsonl(self.config.source)
        elif self.config.input_dir:
            docs = [{"id": doc_id, "text": text} for doc_id, text in _scan_documents(self.config.input_dir)]
        else:
            raise ValueError("IngestionConfig.source or input_dir must be provided.")
        
        # Initialize progress tracking
        logger.info("Starting ingestion pipeline: {} documents to process", len(docs))
        progress = ProgressTracker(
            total=len(docs),
            stage_name="Processing documents",
            total_stages=1,
            current_stage=1,
        )
        
        chunks_records: List[dict] = []
        relations_records: List[dict] = []
        for doc in docs:
            doc_id = doc.get("id") or doc.get("document_id") or f"doc-{len(chunks_records)}"
            text = _normalize_text(doc.get("text", ""))
            relation_cache_key = _relation_cache_key(doc_id, text, self.config)
            cached_relations = None
            if self.config.cache_enabled and not force:
                cached_relations = _load_cached_relations(self.cache_dir, relation_cache_key)
            doc_relations: List[dict] = []
            for idx, chunk in enumerate(_chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)):
                chunk_id = f"{doc_id}::chunk-{idx}"
                chunks_records.append({"id": chunk_id, "document_id": doc_id, "text": chunk})
                if cached_relations is None:
                    chunk_relations, parse_payload = self.extractor.extract(doc_id, chunk_id, chunk)
                    chunk_relations = _apply_relation_classifier(
                        chunk_relations, self.classifier, self.config.relation_threshold
                    )
                    relations_records.extend(chunk_relations)
                    doc_relations.extend(chunk_relations)
                    if parse_payload:
                        _store_cached_parse(self.cache_dir / "parses", chunk_id, parse_payload)
                else:
                    chunk_cached = [rel for rel in cached_relations if rel.get("chunk_id") == chunk_id]
                    relations_records.extend(chunk_cached)
                    doc_relations.extend(chunk_cached)
            if cached_relations is None and self.config.cache_enabled:
                _store_cached_relations(self.cache_dir, relation_cache_key, doc_relations)
            
            # Update progress after each document
            progress.update()

        if not chunks_records:
            raise ValueError("Ingestion pipeline produced zero chunks.")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.cache_dir / "chunks.jsonl"
        relations_path = self.cache_dir / "relations.jsonl"
        _write_jsonl(chunks_path, chunks_records)
        _write_jsonl(relations_path, relations_records)
        duration = max(perf_counter() - start, 1e-9)
        docs_per_sec = len(docs) / duration if docs else 0.0
        logger.info(
            "Ingestion complete: {} docs ({} chunks, {} relations) in {} ({:.2f} docs/sec)",
            len(docs),
            len(chunks_records),
            len(relations_records),
            _format_duration(duration),
            docs_per_sec,
        )
        return IngestionResult(
            chunks_path=chunks_path,
            relations_path=relations_path,
            cache_dir=self.cache_dir,
            output_dir=self.output_dir,
            document_count=len(docs),
            chunk_count=len(chunks_records),
            relation_count=len(relations_records),
        )


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_ingest_config(path: Path) -> IngestionConfig:
    data = json.loads(path.read_text()) if path.suffix == ".json" else None
    if data is None:
        import yaml

        data = yaml.safe_load(path.read_text())
    if "ingest" in data:
        data = data["ingest"]
    return IngestionConfig(**data)


CHUNK_ID_TEMPLATE = "{doc_id}-chunk-{idx}"


def run_ingestion(cfg: IngestionConfig, force: bool = False) -> IngestionResult:
    if not cfg.input_dir and not cfg.source:
        raise ValueError("Either input_dir or source must be provided for ingestion.")
    output_dir = cfg.output_dir
    cache_dir = (cfg.cache_dir or output_dir / "cache").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.jsonl"
    relations_path = output_dir / "relations.jsonl"

    chunk_records: List[dict] = []
    relation_records: List[dict] = []
    start = perf_counter()
    parse_cache_dir = cache_dir / "parses" if cfg.parser != "simple" else None
    extractor = _create_relation_extractor(cfg)
    classifier = (
        RelationClassifier(
            cfg.relation_model,
            cfg.relation_max_length
        )
        if cfg.relation_model
        else None
    )
    if cfg.source:
        documents = list(_load_source_documents(cfg.source))
    else:
        if not cfg.input_dir:  # pragma: no cover - guarded by earlier validation
            raise ValueError("input_dir is required when source is not provided.")
        documents = list(_scan_documents(cfg.input_dir))
    
    # Initialize progress tracking
    logger.info("Starting legacy ingestion pipeline: {} documents to process", len(documents))
    progress = ProgressTracker(
        total=len(documents),
        stage_name="Processing documents",
        total_stages=1,
        current_stage=1,
    )
    
    for doc_id, text in documents:
        cache_key = _chunk_cache_key(doc_id, text, cfg)
        cached_chunks = None if force else _load_cached_chunks(cache_dir, cache_key)
        if cached_chunks is None:
            chunk_texts = _chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
            cached_chunks = [
                {
                    "doc_id": doc_id,
                    "chunk_id": CHUNK_ID_TEMPLATE.replace("{doc_id}", doc_id).replace("{idx}", str(idx)),
                    "text": chunk_text,
                    "token_count": len(chunk_text.split()),
                }
                for idx, chunk_text in enumerate(chunk_texts)
            ]
            _store_cached_chunks(cache_dir, cache_key, cached_chunks)

        chunk_records.extend(cached_chunks)
        relation_cache_key = _relation_cache_key(doc_id, text, cfg)
        cached_relations = None if force else _load_cached_relations(cache_dir, relation_cache_key)
        if cached_relations is not None:
            relation_records.extend(cached_relations)
        else:
            doc_relations: List[dict] = []
            for chunk in cached_chunks:
                chunk_relations, parse_payload = extractor.extract(
                    doc_id=doc_id, chunk_id=chunk["chunk_id"], text=chunk["text"]
                )
                chunk_relations = _apply_relation_classifier(chunk_relations, classifier, cfg.relation_threshold)
                doc_relations.extend(chunk_relations)
                if parse_cache_dir and parse_payload:
                    _store_cached_parse(parse_cache_dir, chunk["chunk_id"], parse_payload)
            relation_records.extend(doc_relations)
            _store_cached_relations(cache_dir, relation_cache_key, doc_relations)
        
        # Update progress after each document
        progress.update()

    _write_jsonl(chunks_path, chunk_records)
    _write_jsonl(relations_path, relation_records)
    duration = max(perf_counter() - start, 1e-9)
    logger.info(
        "Legacy ingestion complete: {} docs ({} chunks, {} relations) in {} ({:.2f} docs/sec)",
        len(documents),
        len(chunk_records),
        len(relation_records),
        _format_duration(duration),
        (len(documents) / duration) if documents else 0.0,
    )

    return IngestionResult(
        chunks_path=chunks_path,
        relations_path=relations_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        document_count=len(documents),
        chunk_count=len(chunk_records),
        relation_count=len(relation_records),
    )


def validate_ingestion_result(result: IngestionResult) -> None:
    if not result.chunks_path.exists():
        raise ValueError("Chunks file missing at {}".format(result.chunks_path))
    if not result.relations_path.exists():
        raise ValueError("Relations file missing at {}".format(result.relations_path))
    if result.chunk_count <= 0:
        raise ValueError("Ingestion produced zero chunks")
    if result.document_count <= 0:
        raise ValueError("Ingestion processed zero documents")

    chunk_records = list(_iter_jsonl(result.chunks_path))
    if len(chunk_records) != result.chunk_count:
        raise ValueError(
            f"Chunk count mismatch: expected {result.chunk_count}, found {len(chunk_records)} in {result.chunks_path}"
        )
    required_chunk_fields = {"id", "document_id", "text"}
    for idx, record in enumerate(chunk_records):
        missing = required_chunk_fields - record.keys()
        if missing:
            raise ValueError(f"Chunk record #{idx} missing fields: {', '.join(sorted(missing))}")

    relation_records = list(_iter_jsonl(result.relations_path))
    if relation_records and len(relation_records) != result.relation_count:
        raise ValueError(
            f"Relation count mismatch: expected {result.relation_count}, found {len(relation_records)} "
            f"in {result.relations_path}"
        )
    required_rel_fields = {"doc_id", "chunk_id", "subject", "predicate", "object"}
    for idx, record in enumerate(relation_records):
        missing = required_rel_fields - record.keys()
        if missing:
            raise ValueError(f"Relation record #{idx} missing fields: {', '.join(sorted(missing))}")


def summarize_ingestion(result: IngestionResult) -> Dict[str, Any]:
    """Generate ingestion statistics (documents, chunks, relations, label histogram)."""

    chunks = list(_iter_jsonl(result.chunks_path))
    relations = list(_iter_jsonl(result.relations_path))
    labels = Counter(rel.get("predicate", "unknown") for rel in relations)
    return {
        "documents": result.document_count,
        "chunks": len(chunks),
        "relations": len(relations),
        "labels": dict(sorted(labels.items(), key=lambda item: item[0])),
    }


def validate_with_schema(
    result: IngestionResult,
    chunk_schema_path: Optional[Path],
    relation_schema_path: Optional[Path],
) -> None:
    chunk_validator = _load_validator(chunk_schema_path) if chunk_schema_path else None
    relation_validator = _load_validator(relation_schema_path) if relation_schema_path else None

    if chunk_validator:
        for idx, record in enumerate(_iter_jsonl(result.chunks_path)):
            errors = list(chunk_validator.iter_errors(record))
            if errors:
                raise ValueError(f"Chunk schema validation failed at record #{idx}: {errors[0].message}")
    if relation_validator:
        for idx, record in enumerate(_iter_jsonl(result.relations_path)):
            errors = list(relation_validator.iter_errors(record))
            if errors:
                raise ValueError(f"Relation schema validation failed at record #{idx}: {errors[0].message}")


def _load_validator(schema_path: Path) -> Draft7Validator:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft7Validator(schema)


def _scan_documents(input_dir: Path) -> Iterable[tuple[str, str]]:
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".json"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        yield path.stem, text


def _load_source_documents(source: Path) -> List[tuple[str, str]]:
    records: List[Any]
    if source.suffix.lower() == ".jsonl":
        records = list(_iter_jsonl(source))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            candidate = payload.get("documents") or payload.get("data")
            if candidate is not None:
                payload = candidate
            else:
                payload = [payload]
        if not isinstance(payload, list):
            payload = [payload]
        records = payload

    documents: List[tuple[str, str]] = []
    for idx, item in enumerate(records):
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("body") or ""
            if not text:
                continue
            doc_identifier = item.get("id") or item.get("title") or f"doc-{idx}"
        else:
            text = str(item)
            doc_identifier = f"doc-{idx}"
        slug = _slugify(str(doc_identifier))
        documents.append((slug or f"doc-{idx}", _normalize_text(text)))

    if not documents:
        raise ValueError(f"No documents found in {source}")
    return documents


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return slug.strip("-")


def _chunk_cache_key(doc_id: str, text: str, cfg: IngestionConfig) -> str:
    hasher = hashlib.sha256()
    hasher.update(doc_id.encode("utf-8"))
    hasher.update(str(cfg.chunk_size).encode("utf-8"))
    hasher.update(str(cfg.chunk_overlap).encode("utf-8"))
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def _load_cached_chunks(cache_dir: Path, cache_key: str) -> Optional[List[dict]]:
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _store_cached_chunks(cache_dir: Path, cache_key: str, chunks: List[dict]) -> None:
    path = cache_dir / f"{cache_key}.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)


def _relation_cache_key(doc_id: str, text: str, cfg: IngestionConfig) -> str:
    hasher = hashlib.sha256()
    hasher.update(doc_id.encode("utf-8"))
    hasher.update(text.encode("utf-8"))
    hasher.update(cfg.parser.encode("utf-8"))
    hasher.update(cfg.parser_model.encode("utf-8"))
    hasher.update(cfg.language.encode("utf-8"))
    hasher.update(str(cfg.relation_threshold).encode("utf-8"))
    hasher.update(str(cfg.relation_model or "").encode("utf-8"))
    return hasher.hexdigest()


def _load_cached_relations(cache_dir: Path, cache_key: str) -> Optional[List[dict]]:
    path = cache_dir / "relations" / f"{cache_key}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _store_cached_relations(cache_dir: Path, cache_key: str, relations: List[dict]) -> None:
    target_dir = cache_dir / "relations"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{cache_key}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(relations, handle, ensure_ascii=False, indent=2)


def _store_cached_parse(cache_dir: Path, chunk_id: str, payload: dict) -> None:
    if not payload:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
    path = cache_dir / f"{safe_name}.json"
    record = {"chunk_id": chunk_id, "payload": payload}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)


def load_cached_parse(cache_dir: Path, chunk_id: str) -> Optional[dict]:
    safe_name = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
    path = cache_dir / f"{safe_name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "payload" in data:
        return data["payload"]
    return data

