---
title: Training Data Guide
description: Understanding and preparing training data for Sphana models
category: Sphana Trainer
---

# Training Data Guide

This guide explains what training data you need to provide to the `ingest` command and how it processes your documents.

## Overview

The `ingest` command is the first step in the Sphana training pipeline. It transforms raw documents into structured training data that can be used to train embedding, relation extraction, GNN, and LLM models.

**What it does**:
1. Reads raw documents from various sources
2. Chunks documents into manageable pieces
3. Extracts entity-relation triples
4. Generates training artifacts for downstream tasks

## Input Data Formats

The ingestion pipeline accepts documents in several formats:

### 1. Plain Text Files (`.txt`, `.md`)

Store each document as a separate file in a directory:

```
samples/my-docs/
├── document1.txt
├── document2.md
└── document3.txt
```

**Configuration**:
```yaml
ingest:
  input_dir: "samples/my-docs"
  artifact_root: "target/ingest"
  chunk_size: 512
  chunk_overlap: 50
```

**Pros**: Simple to create, human-readable  
**Cons**: No metadata support

### 2. JSONL Format (`.jsonl`)

Each line contains a JSON object representing one document:

```jsonl
{"id": 1, "title": "Transfer Learning", "text": "Transfer learning (TL) is a technique...", "source": "wikipedia"}
{"id": 2, "title": "Neural Networks", "text": "Neural networks are...", "source": "internal"}
{"id": 3, "title": "RAG Systems", "text": "Retrieval-Augmented Generation...", "source": "arxiv"}
```

**Required fields**:
- `text` or `content` or `body` - The document content
- `id` (optional) - Document identifier (auto-generated from line number if missing)
- `title` (optional) - Document title
- `source` (optional) - Source metadata

**Configuration**:
```yaml
ingest:
  source: "samples/wiki-docs/large/docs.jsonl"
  artifact_root: "target/ingest"
  chunk_size: 512
  chunk_overlap: 50
```

**Pros**: Supports metadata, efficient for large datasets  
**Cons**: Less human-readable

### 3. JSON Format (`.json`)

Single JSON file with an array of documents:

```json
{
  "documents": [
    {
      "id": "doc1",
      "text": "Machine learning is a subset of artificial intelligence...",
      "metadata": {
        "author": "John Doe",
        "date": "2024-01-15"
      }
    },
    {
      "id": "doc2",
      "text": "Deep learning uses neural networks with multiple layers...",
      "metadata": {
        "author": "Jane Smith",
        "date": "2024-02-20"
      }
    }
  ]
}
```

**Configuration**:
```yaml
ingest:
  source: "samples/documents.json"
  artifact_root: "target/ingest"
  chunk_size: 512
```

**Pros**: Structured metadata, easy to validate  
**Cons**: Must load entire file into memory

## Document Requirements

### Content Quality

For best results, your documents should:

✅ **Be coherent and well-structured** - Complete sentences and paragraphs  
✅ **Contain factual information** - Entities and relationships  
✅ **Have sufficient length** - At least 100-200 words per document  
✅ **Use proper grammar** - Improves relation extraction  
✅ **Be domain-relevant** - Match your target use case

❌ **Avoid**:
- Code snippets without context
- Tables or structured data without descriptions
- Extremely short fragments
- Non-textual content (images, charts)

### Document Length

- **Minimum**: ~50 words (after chunking)
- **Optimal**: 200-2000 words per document
- **Maximum**: No hard limit (will be chunked)

### Character Encoding

- **Required**: UTF-8 encoding
- Non-ASCII characters are supported (Unicode)

## Processing Pipeline

The ingest command processes documents through several stages:

```
Input Documents
      │
      ▼
┌─────────────┐
│  Normalize  │  Remove extra whitespace
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Chunking  │  Split into fixed-size pieces
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Relation   │  Extract entity-relation triples
│ Extraction  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Output    │  chunks.jsonl + relations.jsonl
└─────────────┘
```

### 1. Text Normalization

- Collapses multiple whitespace into single spaces
- Preserves paragraph structure
- Handles UTF-8 characters

### 2. Chunking

Documents are split into overlapping chunks:

**Parameters**:
- `chunk_size`: Number of words per chunk (default: 512)
- `chunk_overlap`: Overlapping words between chunks (default: 50)

**Example**:
```
Document: "Machine learning is powerful. Neural networks learn patterns. Deep learning uses layers."

With chunk_size=8, chunk_overlap=2:

Chunk 0: "Machine learning is powerful. Neural networks learn patterns."
Chunk 1: "learn patterns. Deep learning uses layers."
         ^^^^^^^^^^^^^ (overlap)
```

**Why overlap?** Preserves context at chunk boundaries and improves relation extraction.

### 3. Relation Extraction

Three extraction methods are available:

#### a) **Simple (Regex-based)**

```yaml
ingest:
  parser: "simple"
  relation_threshold: 0.7
```

- Fast, no dependencies
- Pattern: `Subject <verb phrase> Object`
- Example: "Alice built NRDB" → (Alice, built, NRDB)
- **Use for**: Quick prototyping, smoke tests

#### b) **spaCy (Dependency parsing)**

```yaml
ingest:
  parser: "spacy"
  parser_model: "en_core_web_sm"
  relation_threshold: 0.7
```

- Linguistically accurate
- Uses dependency trees
- Extracts subject-verb-object patterns
- **Use for**: English text, production quality

**Before using**, download the model:
```bash
python -m sphana_trainer.cli ingest-cache-models \
    --spacy-model en_core_web_sm
```

#### c) **Stanza (Multi-language)**

```yaml
ingest:
  parser: "stanza"
  language: "en"
  relation_threshold: 0.7
```

- Supports 60+ languages
- Neural dependency parsing
- Slower but more accurate
- **Use for**: Non-English text, research

**Before using**, download the model:
```bash
python -m sphana_trainer.cli ingest-cache-models \
    --stanza-lang en
```

#### d) **HuggingFace Classifier (Optional)**

Refine relation labels using a trained classifier:

```yaml
ingest:
  relation_model: "FacebookAI/bart-large-mnli"
  relation_threshold: 0.8
```

- Classifies extracted relations into types
- Filters low-confidence relations
- **Use for**: Domain-specific relation types

## Output Files

The ingestion produces three outputs:

### 1. `chunks.jsonl`

Document chunks ready for embedding training:

```jsonl
{"id": "doc1-chunk-0", "document_id": "doc1", "text": "Transfer learning is...", "token_count": 487}
{"id": "doc1-chunk-1", "document_id": "doc1", "text": "...a technique in machine learning...", "token_count": 512}
{"id": "doc2-chunk-0", "document_id": "doc2", "text": "Neural networks are...", "token_count": 423}
```

**Fields**:
- `id`: Unique chunk identifier
- `document_id`: Source document ID
- `text`: Chunk content
- `token_count`: Word count

### 2. `relations.jsonl`

Extracted entity-relation triples:

```jsonl
{"doc_id": "doc1", "chunk_id": "doc1-chunk-0", "subject": "Transfer learning", "predicate": "is_a", "object": "technique", "confidence": 0.92, "sentence": "Transfer learning is a technique..."}
{"doc_id": "doc1", "chunk_id": "doc1-chunk-1", "subject": "knowledge", "predicate": "learned_from", "object": "task", "confidence": 0.87, "sentence": "knowledge learned from a task..."}
```

**Fields**:
- `doc_id`: Source document
- `chunk_id`: Source chunk
- `subject`: Head entity
- `predicate`: Relation type
- `object`: Tail entity
- `confidence`: Extraction confidence (0-1)
- `sentence`: Source sentence

### 3. `cache/`

Cached intermediate results for faster re-runs:

```
target/ingest/cache/
├── chunks/
│   └── <hash>.json
├── relations/
│   └── <hash>.json
└── parse/
    └── <chunk_id>.json
```

## Configuration Reference

Complete YAML configuration options:

```yaml
ingest:
  # Input source (choose one)
  source: "samples/docs.jsonl"           # Single file (JSONL or JSON)
  # OR
  input_dir: "samples/my-docs"           # Directory of text files
  
  # Output
  artifact_root: "target/ingest"         # Output directory
  
  # Chunking
  chunk_size: 512                        # Words per chunk
  chunk_overlap: 50                      # Overlapping words
  
  # Relation extraction
  parser: "spacy"                        # "simple", "spacy", or "stanza"
  parser_model: "en_core_web_sm"         # spaCy model (if parser=spacy)
  language: "en"                         # Language code (if parser=stanza)
  relation_threshold: 0.7                # Confidence threshold (0-1)
  
  # Optional: Relation classification
  relation_model: null                   # HuggingFace model for relation typing
  relation_calibration: null             # Calibration file for confidence scores
```

## Usage Examples

### Example 1: Wikipedia Articles (JSONL)

```bash
# 1. Prepare data (JSONL format)
cat > wiki-articles.jsonl << EOF
{"id": 1, "title": "Machine Learning", "text": "Machine learning is..."}
{"id": 2, "title": "Deep Learning", "text": "Deep learning is..."}
EOF

# 2. Create config
cat > configs/ingest/wiki.yaml << EOF
ingest:
  source: "wiki-articles.jsonl"
  artifact_root: "target/ingest"
  parser: "spacy"
  parser_model: "en_core_web_sm"
  chunk_size: 512
  chunk_overlap: 50
  relation_threshold: 0.75
EOF

# 3. Cache models
python -m sphana_trainer.cli ingest-cache-models --spacy-model en_core_web_sm

# 4. Run ingestion
python -m sphana_trainer.cli ingest --config configs/ingest/wiki.yaml
```

### Example 2: Company Documentation (Text Files)

```bash
# 1. Organize files
mkdir -p samples/company-docs
cp /path/to/docs/*.md samples/company-docs/

# 2. Create config
cat > configs/ingest/company.yaml << EOF
ingest:
  input_dir: "samples/company-docs"
  artifact_root: "target/ingest"
  parser: "simple"
  chunk_size: 256
  chunk_overlap: 30
EOF

# 3. Run ingestion (no model caching needed for 'simple')
python -m sphana_trainer.cli ingest --config configs/ingest/company.yaml
```

### Example 3: Multi-language Content

```bash
# 1. Create config for French documents
cat > configs/ingest/french.yaml << EOF
ingest:
  source: "samples/french-docs.jsonl"
  artifact_root: "target/ingest"
  parser: "stanza"
  language: "fr"
  chunk_size: 512
  chunk_overlap: 50
  relation_threshold: 0.7
EOF

# 2. Cache Stanza model
python -m sphana_trainer.cli ingest-cache-models --stanza-lang fr

# 3. Run ingestion
python -m sphana_trainer.cli ingest --config configs/ingest/french.yaml
```

## Validation

After ingestion, validate the output:

```bash
python -m sphana_trainer.cli ingest-validate \
    --config configs/ingest/wiki.yaml \
    --stats \
    --chunks-schema src/sphana_trainer/schemas/ingestion/chunks.schema.json \
    --relations-schema src/sphana_trainer/schemas/ingestion/relations.schema.json
```

**Checks**:
- ✅ File existence (`chunks.jsonl`, `relations.jsonl`)
- ✅ JSON schema compliance
- ✅ Record counts match metadata
- ✅ Required fields present
- 📊 Statistics (chunk length, relation types, etc.)

## Best Practices

### 1. Start Small

Begin with a small dataset (10-50 documents) to:
- Test your configuration
- Verify output quality
- Tune parameters

### 2. Choose the Right Parser

| Parser | Speed | Accuracy | Languages | Use Case |
|--------|-------|----------|-----------|----------|
| `simple` | ⚡️ Fast | ⭐️ Basic | English | Prototyping |
| `spacy` | ⚡️⚡️ Medium | ⭐️⭐️⭐️ Good | 20+ | Production |
| `stanza` | ⚡️ Slow | ⭐️⭐️⭐️⭐️ Best | 60+ | Research |

### 3. Tune Chunk Size

- **Smaller chunks** (128-256): Better for short facts, faster training
- **Larger chunks** (512-1024): Better context, richer embeddings
- **Rule of thumb**: Match your expected query length

### 4. Monitor Cache Usage

```bash
# Check cache size
du -sh target/ingest/cache

# Clear cache to re-process
rm -rf target/ingest/cache
# OR use --force flag
python -m sphana_trainer.cli ingest --config configs/ingest/base.yaml --force
```

### 5. Handle Large Datasets

For datasets > 10K documents:

- Use JSONL format (streaming)
- Process in batches
- Monitor memory usage
- Use `--force` sparingly (cache saves time)

## Troubleshooting

### Issue: "No documents found"

**Cause**: Input path incorrect or files empty

**Solution**:
```bash
# Check files exist
ls -la samples/my-docs/

# Verify file content
head samples/my-docs/doc1.txt

# Check JSONL format
cat samples/docs.jsonl | jq .
```

### Issue: "Zero relations extracted"

**Cause**: Parser not finding relationships, threshold too high

**Solution**:
- Lower `relation_threshold` (try 0.5)
- Try a different parser (`spacy` vs `stanza`)
- Verify documents contain subject-verb-object patterns
- Check example output: `cat target/ingest/relations.jsonl | head`

### Issue: "Module not found: spacy"

**Cause**: spaCy model not downloaded

**Solution**:
```bash
python -m sphana_trainer.cli ingest-cache-models --spacy-model en_core_web_sm
```

### Issue: "Out of memory"

**Cause**: Large dataset, insufficient RAM

**Solution**:
- Use `source` (JSONL) instead of `input_dir` for streaming
- Reduce `chunk_size`
- Process in batches
- Increase system memory

## Next Steps

After successful ingestion, proceed to training:

1. **Embedding Model**: [`train embedding`](./trainer-cli#train-embedding)
2. **Relation Model**: [`train relation`](./trainer-cli#train-relation)
3. **GNN Model**: [`train gnn`](./trainer-cli#train-gnn)
4. **LLM Model**: [`train llm`](./trainer-cli#train-llm)

See the [Training Workflow](./trainer-workflow) for the complete pipeline.

## Additional Resources

- [CLI Reference](./trainer-cli) - All ingestion commands
- [Configuration Guide](./trainer-config) - YAML configuration details
- [Training Workflow](./trainer-workflow) - End-to-end pipeline
- Sample data: `services/Sphana.Trainer/samples/wiki-docs/`

