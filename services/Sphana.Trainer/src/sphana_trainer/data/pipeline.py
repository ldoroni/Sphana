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
    import gzip
    
    if not path.exists():
        return []
    
    # Check if file is gzip compressed
    is_gzipped = path.suffix.lower() == '.gz' or path.name.endswith('.jsonl.gz')
    
    if is_gzipped:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _write_jsonl(records: List[dict], output_path: Path, compress: bool = False) -> None:
    """Write records to JSONL file, optionally compressed."""
    import gzip
    
    if compress and not str(output_path).endswith('.gz'):
        output_path = Path(str(output_path) + '.gz')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    else:
        with output_path.open('w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')


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
    chunks_output_dir: Path
    relations_output_dir: Path
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
        self.chunks_output_dir = config.chunks_output_dir.expanduser().resolve()
        self.relations_output_dir = config.relations_output_dir.expanduser().resolve()
        self.cache_dir = (config.cache_dir or self.chunks_output_dir.parent / "cache").expanduser().resolve()
        self.chunks_output_dir.mkdir(parents=True, exist_ok=True)
        self.relations_output_dir.mkdir(parents=True, exist_ok=True)
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
                    chunks_output_dir=self.chunks_output_dir,
                    relations_output_dir=self.relations_output_dir,
                )

        # Load documents from source (supports glob patterns, returns domain info)
        docs_with_domain = [
            {"id": doc_id, "text": text, "domain": domain} 
            for doc_id, text, domain in _load_documents_from_source(self.config.source)
        ]
        
        # Group documents by domain
        from collections import defaultdict
        domain_docs = defaultdict(list)
        for doc in docs_with_domain:
            domain_docs[doc['domain']].append(doc)
        
        # Determine if we need per-domain outputs (multiple domains found)
        use_domain_suffix = len(domain_docs) > 1
        
        # Initialize progress tracking
        total_docs = len(docs_with_domain)
        logger.info("Starting ingestion pipeline: {} documents to process across {} domain(s)", 
                   total_docs, len(domain_docs))
        progress = ProgressTracker(
            total=total_docs,
            stage_name="Processing documents",
            total_stages=1,
            current_stage=1,
            log_interval=self.config.progress_log_interval,
        )
        
        all_chunks_records: List[dict] = []
        all_relations_records: List[dict] = []
        
        # Process each domain
        for domain, docs in domain_docs.items():
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
            
            # Write domain-specific files
            if use_domain_suffix:
                # Multi-domain: use subdirectories within output dirs
                self.chunks_output_dir.mkdir(parents=True, exist_ok=True)
                self.relations_output_dir.mkdir(parents=True, exist_ok=True)
                
                chunks_filename = f"{domain}.jsonl"
                relations_filename = f"{domain}.jsonl"
                chunks_path = self.chunks_output_dir / chunks_filename
                relations_path = self.relations_output_dir / relations_filename
            else:
                # Single domain - use traditional filenames
                self.chunks_output_dir.mkdir(parents=True, exist_ok=True)
                self.relations_output_dir.mkdir(parents=True, exist_ok=True)
                
                chunks_filename = "chunks.jsonl"
                relations_filename = "relations.jsonl"
                chunks_path = self.chunks_output_dir / chunks_filename
                relations_path = self.relations_output_dir / relations_filename
            
            _write_jsonl(chunks_records, chunks_path, self.config.output_compressed)
            _write_jsonl(relations_records, relations_path, self.config.output_compressed)
            
            all_chunks_records.extend(chunks_records)
            all_relations_records.extend(relations_records)
            
            logger.info(f"Domain '{domain}': {len(docs)} docs, {len(chunks_records)} chunks, {len(relations_records)} relations")

        if not all_chunks_records:
            raise ValueError("Ingestion pipeline produced zero chunks.")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_chunks_path = self.cache_dir / "chunks.jsonl"
        cache_relations_path = self.cache_dir / "relations.jsonl"
        _write_jsonl(all_chunks_records, cache_chunks_path, False)
        _write_jsonl(all_relations_records, cache_relations_path, False)
        
        duration = max(perf_counter() - start, 1e-9)
        docs_per_sec = total_docs / duration if total_docs else 0.0
        logger.info(
            "Ingestion complete: {} docs ({} chunks, {} relations) in {} ({:.2f} docs/sec)",
            total_docs,
            len(all_chunks_records),
            len(all_relations_records),
            _format_duration(duration),
            docs_per_sec,
        )
        return IngestionResult(
            chunks_path=cache_chunks_path,
            relations_path=cache_relations_path,
            cache_dir=self.cache_dir,
            chunks_output_dir=self.chunks_output_dir,
            relations_output_dir=self.relations_output_dir,
            document_count=total_docs,
            chunk_count=len(all_chunks_records),
            relation_count=len(all_relations_records),
        )


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
    # Load documents from source (supports glob patterns)
    chunks_output_dir = cfg.chunks_output_dir.expanduser().resolve()
    relations_output_dir = cfg.relations_output_dir.expanduser().resolve()
    cache_dir = (cfg.cache_dir or chunks_output_dir.parent / "cache").expanduser().resolve()
    chunks_output_dir.mkdir(parents=True, exist_ok=True)
    relations_output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

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
    
    # Load documents using the new glob-aware function (with domain info)
    docs_with_domain = list(_load_documents_from_source(cfg.source))
    
    # Group documents by domain
    from collections import defaultdict
    domain_docs = defaultdict(list)
    for doc_id, text, domain in docs_with_domain:
        domain_docs[domain].append((doc_id, text))
    
    # Determine if we need per-domain outputs (multiple domains found)
    use_domain_suffix = len(domain_docs) > 1
    
    # Initialize progress tracking
    total_docs = len(docs_with_domain)
    logger.info("Starting legacy ingestion pipeline: {} documents to process across {} domain(s)", 
               total_docs, len(domain_docs))
    progress = ProgressTracker(
        total=total_docs,
        stage_name="Processing documents",
        total_stages=1,
        current_stage=1,
        log_interval=cfg.progress_log_interval,
    )
    
    # Process each domain
    for domain, documents in domain_docs.items():
        domain_chunks: List[dict] = []
        domain_relations: List[dict] = []
        
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

            domain_chunks.extend(cached_chunks)
            relation_cache_key = _relation_cache_key(doc_id, text, cfg)
            cached_relations = None if force else _load_cached_relations(cache_dir, relation_cache_key)
            if cached_relations is not None:
                domain_relations.extend(cached_relations)
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
                domain_relations.extend(doc_relations)
                _store_cached_relations(cache_dir, relation_cache_key, doc_relations)
            
            # Update progress after each document
            progress.update()
        
        # Write domain-specific files
        if use_domain_suffix:
            # Multi-domain: files in output dirs
            chunks_output_dir.mkdir(parents=True, exist_ok=True)
            relations_output_dir.mkdir(parents=True, exist_ok=True)
            
            chunks_filename = f"{domain}.jsonl"
            relations_filename = f"{domain}.jsonl"
            chunks_path = chunks_output_dir / chunks_filename
            relations_path = relations_output_dir / relations_filename
        else:
            # Single domain - use traditional filenames
            chunks_output_dir.mkdir(parents=True, exist_ok=True)
            relations_output_dir.mkdir(parents=True, exist_ok=True)
            
            chunks_filename = "chunks.jsonl"
            relations_filename = "relations.jsonl"
            chunks_path = chunks_output_dir / chunks_filename
            relations_path = relations_output_dir / relations_filename
        
        _write_jsonl(domain_chunks, chunks_path, cfg.output_compressed)
        _write_jsonl(domain_relations, relations_path, cfg.output_compressed)
        
        chunk_records.extend(domain_chunks)
        relation_records.extend(domain_relations)
        
        logger.info(f"Domain '{domain}': {len(documents)} docs, {len(domain_chunks)} chunks, {len(domain_relations)} relations")

    duration = max(perf_counter() - start, 1e-9)
    logger.info(
        "Legacy ingestion complete: {} docs ({} chunks, {} relations) in {} ({:.2f} docs/sec)",
        total_docs,
        len(chunk_records),
        len(relation_records),
        _format_duration(duration),
        (total_docs / duration) if total_docs else 0.0,
    )

    # Return paths to cache (for backwards compatibility)
    return IngestionResult(
        chunks_path=chunks_output_dir / ("chunks.jsonl" if not use_domain_suffix else f"{list(domain_docs.keys())[0]}.jsonl"),
        relations_path=relations_output_dir / ("relations.jsonl" if not use_domain_suffix else f"{list(domain_docs.keys())[0]}.jsonl"),
        cache_dir=cache_dir,
        chunks_output_dir=chunks_output_dir,
        relations_output_dir=relations_output_dir,
        document_count=total_docs,
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


def _resolve_source_files(source_pattern: str) -> List[Path]:
    """
    Resolve a source pattern to a list of file paths.
    Supports:
    - Single file: 'path/to/file.jsonl' or 'path/to/file.jsonl.gz'
    - Glob patterns: 'path/*.jsonl.gz', 'path/**/*.txt'
    """
    from glob import glob
    from pathlib import Path
    
    # Convert to Path for normalization
    source_path = Path(source_pattern).expanduser()
    
    # Check if it's a literal file (no glob characters)
    if not any(char in str(source_path) for char in ['*', '?', '[', ']']):
        if source_path.exists() and source_path.is_file():
            return [source_path]
        else:
            raise ValueError(f"Source file not found: {source_pattern}")
    
    # It's a glob pattern - resolve it
    resolved_paths = [Path(p) for p in glob(str(source_path), recursive=True)]
    resolved_files = [p for p in resolved_paths if p.is_file()]
    
    if not resolved_files:
        raise ValueError(f"No files matched pattern: {source_pattern}")
    
    return sorted(resolved_files)


def _load_documents_from_source(source_pattern: str) -> List[tuple[str, str, str]]:
    """
    Load documents from source pattern (file or glob).
    Returns list of (doc_id, text, domain) tuples.
    Domain is derived from the source filename.
    """
    import gzip
    
    files = _resolve_source_files(source_pattern)
    documents: List[tuple[str, str, str]] = []
    
    for file_path in files:
        # Extract domain from filename
        domain = file_path.stem
        # For .jsonl.gz, remove the .jsonl part too
        if domain.endswith('.jsonl'):
            domain = domain[:-6]
        
        # Handle different file types
        suffix = file_path.suffix.lower()
        name_lower = file_path.name.lower()
        
        # JSONL files (compressed or not)
        if suffix == ".jsonl" or name_lower.endswith('.jsonl.gz'):
            records = list(_iter_jsonl(file_path))
            for idx, item in enumerate(records):
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("body") or ""
                    if not text:
                        continue
                    doc_identifier = item.get("id") or item.get("title") or f"{file_path.stem}-{idx}"
                else:
                    text = str(item)
                    doc_identifier = f"{file_path.stem}-{idx}"
                slug = _slugify(str(doc_identifier))
                documents.append((slug or f"{file_path.stem}-{idx}", _normalize_text(text), domain))
        
        # Plain text files
        elif suffix in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                documents.append((file_path.stem, _normalize_text(text), domain))
        
        # JSON files (single document or array)
        elif suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                candidate = payload.get("documents") or payload.get("data")
                if candidate is not None:
                    payload = candidate
                else:
                    payload = [payload]
            if not isinstance(payload, list):
                payload = [payload]
            
            for idx, item in enumerate(payload):
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("body") or ""
                    if not text:
                        continue
                    doc_identifier = item.get("id") or item.get("title") or f"{file_path.stem}-{idx}"
                else:
                    text = str(item)
                    doc_identifier = f"{file_path.stem}-{idx}"
                slug = _slugify(str(doc_identifier))
                documents.append((slug or f"{file_path.stem}-{idx}", _normalize_text(text), domain))
    
    if not documents:
        raise ValueError(f"No documents found in source: {source_pattern}")
    
    return documents


def _load_source_documents(source: Path) -> List[tuple[str, str]]:
    import gzip
    
    records: List[Any]
    # Check if file is JSONL (compressed or not)
    is_jsonl = source.suffix.lower() == ".jsonl" or source.name.endswith('.jsonl.gz')
    
    if is_jsonl:
        records = list(_iter_jsonl(source))
    else:
        # Handle regular JSON files (not compressed)
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

