---
title: Training Data Guide
description: How to prepare high-quality datasets for the Sphana ingestion pipeline
category: Sphana Trainer
---

# Training Data Guide

This guide focuses on preparing **input data for the `ingest` command**, the first and most critical step in the Sphana training pipeline. The quality of your training data directly impacts the accuracy of all downstream models.

## What is the Ingest Command?

The `ingest` command transforms raw documents into structured training data:

```bash
python -m sphana_trainer.cli ingest --config configs/ingest/wiki.yaml
```

**Input**: Raw text documents  
**Output**: Structured chunks and relations ready for training  
**Purpose**: Foundation for all model training (embedding, relation, GNN)

## Downloading Wikipedia Data

The fastest way to get high-quality training data is to download Wikipedia articles:

```bash
python -m sphana_trainer.cli dataset-download-wiki \
    --titles-dir samples/wiki-titles/large/ \
    --full-content \
    --output samples/wiki-docs/large/ \
    --compress \
    --limit 500000
```

**Key Parameters:**
- `--titles-dir`: Directory with `.txt` files containing Wikipedia titles (one per line)
- `--full-content`: Download complete articles (recommended for accuracy)
- `--output`: Output path (file or directory depending on mode)
- `--output-mode`: Format - `single-jsonl`, `jsonl-per-domain` (default), or `txt-per-doc`
- `--compress`: Compress JSONL files with gzip (saves 70-90% disk space)
- `--limit`: Maximum articles to download

**Example title file** (`samples/wiki-titles/large/01-quantum-physics.txt`):
```
Quantum mechanics
String theory
Particle physics
Wave-particle duality
Quantum entanglement
```

### Output Modes

The download command supports three output modes to organize your data:

#### Mode 1: JSONL per Domain (Default)

**Best for**: Domain-specific training, parallel processing, modular datasets

```bash
python -m sphana_trainer.cli dataset-download-wiki \
    --titles-dir samples/wiki-titles/large/ \
    --output samples/wiki-docs/large/ \
    --compress \
    --full-content
```

**Output structure:**
```
samples/wiki-docs/large/
├── 01-quantum-physics.jsonl.gz
├── 02-organic-chemistry.jsonl.gz
├── 03-molecular-biology.jsonl.gz
└── ...
```

**Advantages:**
- ✅ Organized by domain (from title file names)
- ✅ Easy to train on specific domains
- ✅ Parallel ingestion possible
- ✅ Each file independently compressed

#### Mode 2: Single JSONL

**Best for**: Simple workflows, single-domain datasets, backwards compatibility

```bash
python -m sphana_trainer.cli dataset-download-wiki \
    --titles-dir samples/wiki-titles/large/ \
    --output samples/wiki-docs/large/all-docs.jsonl \
    --output-mode single-jsonl \
    --compress \
    --full-content
```

**Output**: Single file `all-docs.jsonl.gz` with all documents

**Advantages:**
- ✅ Simple single-file output
- ✅ Easy to copy/move
- ✅ Works with existing pipelines

#### Mode 3: Text Files per Document

**Best for**: Human review, git-friendly storage, individual document editing

```bash
python -m sphana_trainer.cli dataset-download-wiki \
    --titles-dir samples/wiki-titles/large/ \
    --output samples/wiki-docs/large/ \
    --output-mode txt-per-doc \
    --full-content
```

**Output structure:**
```
samples/wiki-docs/large/
├── 01-quantum-physics/
│   ├── Quantum_mechanics.txt
│   ├── String_theory.txt
│   └── ...
├── 02-organic-chemistry/
│   ├── Alkane.txt
│   ├── Benzene.txt
│   └── ...
└── ...
```

**Advantages:**
- ✅ Human-readable directory structure
- ✅ Easy to review individual documents
- ✅ Git-friendly (can track changes per file)

### Compression Support

Enable `--compress` to save disk space:

```bash
--compress  # Adds gzip compression to JSONL files
```

**Compression benefits:**
- 💾 **70-90% smaller files** (64K docs: ~5GB → ~500MB)
- 🚀 **Faster network transfer** when copying files
- ✅ **Transparent to ingest** - automatically decompressed
- 🔄 **Per-domain compression** - Each domain file compressed separately

**Note**: Compression only applies to `single-jsonl` and `jsonl-per-domain` modes. Text files (`txt-per-doc`) are not compressed.

### Recommended Dataset Sizes

| Dataset Size | Documents | Use Case | Expected Accuracy | Time to Ingest |
|--------------|-----------|----------|-------------------|----------------|
| **Small** | 100-500 | Testing, prototyping | 60-70% | 5-15 minutes |
| **Medium** | 1,000-10,000 | Development, POC | 75-85% | 1-3 hours |
| **Large** | 50,000-100,000 | Production, high accuracy | 85-95% | 8-16 hours |
| **Very Large** | 500,000+ | Research, maximum accuracy | 90-98% | 3-7 days |

**Recommendation for production**: 50,000-100,000 high-quality Wikipedia articles across diverse domains provide excellent accuracy while keeping training time reasonable.

## Input Data Format

The `ingest` command accepts flexible input via the `source` parameter, which supports:
- Single files (`.jsonl`, `.jsonl.gz`, `.txt`, `.md`, `.json`)
- **Glob patterns** for processing multiple files: `*.jsonl.gz`, `**/*.txt`

### Glob Pattern Support

**Examples:**
```yaml
ingest:
  # Single file
  source: "samples/wiki-docs/file.jsonl.gz"
  
  # All compressed JSONL in directory
  source: "samples/wiki-docs/*.jsonl.gz"
  
  # All text files recursively
  source: "samples/wiki-docs/**/*.txt"
  
  # Process per-domain files from dataset-download-wiki
  source: "samples/wiki-docs/large/*.jsonl.gz"
```

### Format 1: JSONL File (Recommended for Large Datasets)

**File**: One document per line in JSON format (optionally compressed)

```jsonl
{"id": "doc1", "text": "Machine learning is a subset of artificial intelligence...", "domain": "ai"}
{"id": "doc2", "text": "Neural networks are computational models inspired by biological neurons...", "domain": "ai"}
{"id": "doc3", "text": "Deep learning uses multiple layers to progressively extract features...", "domain": "deep-learning"}
```

**Required fields:**
- `text`: The document content (string)

**Optional fields:**
- `id`: Unique document identifier (auto-generated if missing)
- `title`: Document title
- `source`: Source metadata (e.g., "wikipedia", "internal")
- `domain`: Domain/category (automatically added by `dataset-download-wiki`)

**Configuration:**
```yaml
ingest:
  # Single file
  source: "samples/wiki-docs/large/docs.jsonl"
  
  # Compressed file
  source: "samples/wiki-docs/large/docs.jsonl.gz"
  
  # Glob pattern - all .jsonl.gz in directory
  source: "samples/wiki-docs/large/*.jsonl.gz"
  
  # Recursive - all .jsonl.gz files
  source: "samples/wiki-docs/**/*.jsonl.gz"
```

**Advantages:**
- ✅ Efficient for large datasets (streaming)
- ✅ Supports metadata (including domain information)
- ✅ Easy to append new documents
- ✅ Progress tracking works correctly
- ✅ **Supports gzip compression** (`.jsonl.gz` files automatically detected)
- ✅ **Glob patterns** for processing multiple files at once

**Compressed JSONL:**
- Files ending in `.jsonl.gz` are automatically decompressed during ingestion
- No configuration changes needed - works transparently
- Significantly reduces storage requirements (70-90% smaller)

### Format 2: Text Files

**Structure**: Individual `.txt` or `.md` files, selected via glob patterns

**Configuration:**
```yaml
ingest:
  # All text files in a directory
  source: "samples/my-docs/*.txt"
  
  # All markdown files recursively
  source: "samples/my-docs/**/*.md"
  
  # All txt/md files in domain directory
  source: "samples/wiki-docs/quantum-physics/*.txt"
```

**Advantages:**
- ✅ Human-readable directory structure
- ✅ Easy to review individual documents
- ✅ Git-friendly (can track changes per file)
- ✅ **Glob patterns** for flexible file selection

## Building a High-Quality Dataset

### 1. Document Quality Checklist

For maximum accuracy, each document should:

✅ **Content Quality:**
- Complete, coherent sentences
- Factual information with entities and relationships
- Minimum 100-200 words per document
- Proper grammar and punctuation
- Domain-relevant content

✅ **Structure:**
- Clear paragraph breaks
- Logical flow of information
- Well-formed sentences

❌ **Avoid:**
- Code snippets without context
- Raw tables or lists without descriptions
- Very short fragments (< 50 words)
- Non-textual content
- Excessive special characters

### 2. Recommended Dataset Composition

For high accuracy on a specific domain:

**Domain Coverage:**
- **Core concepts**: 40-50% of documents
- **Related concepts**: 30-40% of documents
- **Context/background**: 10-20% of documents

**Example for Medical AI System:**
```
Core (45%): Medical conditions, treatments, procedures
Related (35%): Anatomy, physiology, pharmacology
Context (20%): Medical history, research methodology
```

**Document Distribution:**
- Aim for 500-1,000 documents per major topic
- Include diverse writing styles (encyclopedic, technical, explanatory)
- Balance between breadth (many topics) and depth (detailed coverage)

### 3. Dataset Size Guidelines

| Goal | Minimum Docs | Recommended Docs | Optimal Docs |
|------|--------------|------------------|--------------|
| Proof of Concept | 500 | 2,000 | 5,000 |
| Development/Testing | 5,000 | 20,000 | 50,000 |
| **Production (High Accuracy)** | **20,000** | **64,000** | **100,000+** |
| Research/Maximum Quality | 100,000 | 250,000 | 500,000+ |

**For your 64K Wikipedia dataset** (100 domains × 640 docs avg):
- Excellent balance of breadth and depth
- Expected accuracy: **90-95%** with optimal configuration
- Processing time: **3-5 hours** with GPU

## Optimal Ingest Configuration

For high accuracy with large datasets, use these **recommended settings**:

```yaml
ingest:
  # Input
  source: "samples/wiki-docs/large/docs.jsonl"
  
  # Output
  output_dir: target/ingest
  cache_dir: target/cache
  cache_enabled: true
  
  # Chunking (CRITICAL for accuracy)
  chunk_size: 384          # Larger chunks = better context
  chunk_overlap: 48        # ~12-15% overlap preserves relations
  
  # Parser (CRITICAL for accuracy)
  parser: stanza           # Best accuracy (95-97%)
  language: en
  
  # Relation Extraction
  relation_threshold: 0.20 # Lower = more relations (higher recall)
  
  # Second-stage Classification (OPTIONAL but recommended)
  relation_model: facebook/bart-large-mnli  # Best quality
  relation_max_length: 512
  
  # Progress Logging
  progress_log_interval: 1  # Log every 1% for large datasets
```

### Why These Settings?

| Parameter | Value | Reason |
|-----------|-------|--------|
| `chunk_size` | 384 | Provides rich context for embeddings while staying under 512 token limit |
| `chunk_overlap` | 48 | 12.5% overlap ensures relations spanning boundaries aren't lost |
| `parser` | stanza | State-of-the-art accuracy (95-97%), worth the extra time |
| `relation_threshold` | 0.20 | Low threshold captures more relations; noise filtered later |
| `relation_model` | BART | Best relation classification quality (93-95% accuracy) |
| `relation_max_length` | 512 | Matches BART's optimal sequence length |
| `progress_log_interval` | 1 | Frequent updates for long-running operations |

## Step-by-Step: Building Your Dataset

### Step 1: Choose Your Data Source

**Option A: Download Wikipedia** (Recommended for high accuracy)

```bash
# Create title files for your domains
mkdir -p samples/wiki-titles/my-dataset
echo "Artificial intelligence" >> samples/wiki-titles/my-dataset/ai.txt
echo "Machine learning" >> samples/wiki-titles/my-dataset/ai.txt
echo "Neural network" >> samples/wiki-titles/my-dataset/ai.txt
# ... add 500-1000 titles per domain ...

# Download articles
python -m sphana_trainer.cli dataset-download-wiki \
    --titles-dir samples/wiki-titles/my-dataset/ \
    --full-content \
    --output samples/wiki-docs/my-dataset/docs.jsonl \
    --limit 100000
```

**Option B: Use Your Own Documents**

1. Collect documents (PDFs, web pages, internal docs)
2. Convert to plain text
3. Create JSONL file:

```python
import json

documents = [
    {"id": "doc1", "title": "...", "text": "..."},
    {"id": "doc2", "title": "...", "text": "..."},
]

with open("my-docs.jsonl", "w") as f:
    for doc in documents:
        f.write(json.dumps(doc) + "\n")
```

### Step 2: Install Required Models

```bash
# For Stanza parser (recommended)
python -m sphana_trainer.cli ingest-cache-models --stanza-lang en

# For spaCy parser (alternative)
python -m sphana_trainer.cli ingest-cache-models --spacy-model en_core_web_trf
```

### Step 3: Run Ingestion

```bash
python -m sphana_trainer.cli ingest --config configs/ingest/wiki.yaml
```

**Expected output:**
```
================================================================================
STARTING: Ingestion task
Config: parser=stanza, chunk_size=384, relation_model=facebook/bart-large-mnli
================================================================================
Stanza parser using GPU: cuda
RelationClassifier using GPU: cuda
Starting ingestion pipeline: 64000 documents to process
Stage 1/1: Processing documents | 1% complete | 640/64000 items | Elapsed: 2m 15s | ETA: 3h 35m | Speed: 4.7 items/sec
Stage 1/1: Processing documents | 2% complete | 1280/64000 items | Elapsed: 4m 30s | ETA: 3h 31m | Speed: 4.7 items/sec
...
Ingestion complete: 64000 docs (640000 chunks, 1280000 relations) in 3h 22m (5.3 docs/sec)
================================================================================
COMPLETED: Ingestion task
Results: docs=64000, chunks=640000, relations=1280000, output=target/ingest
================================================================================
```

### Step 4: Validate Output

```bash
# Check output files
ls -lh target/ingest/
# Should see: chunks.jsonl, relations.jsonl

# Inspect sample data
head -n 3 target/ingest/chunks.jsonl
head -n 3 target/ingest/relations.jsonl

# Run validation (optional)
python -m sphana_trainer.cli ingest-validate \
    --config configs/ingest/wiki.yaml \
    --stats
```

## Parser Comparison

| Parser | Accuracy | Speed (64K docs) | GPU Support | Languages | Best For |
|--------|----------|------------------|-------------|-----------|----------|
| **simple** | 40-50% | 30-60 min | ❌ No | English only | Quick testing |
| **spacy** | 75-85% | 2-4 hours | ❌ No | 20+ | Good balance |
| **stanza** | **90-97%** | **3-5 hours** | ✅ **Yes** | **60+** | **Maximum accuracy** |

**For production and high accuracy: Use `stanza` with GPU.**

## Common Issues and Solutions

### Issue: "Not enough relations extracted"

**Symptoms**: `relations.jsonl` has very few entries

**Solutions:**
1. Lower `relation_threshold` (try 0.15-0.25)
2. Use better parser (`stanza` instead of `simple`)
3. Check document quality (should contain clear subject-verb-object patterns)
4. Verify documents aren't just lists or tables

### Issue: "Ingestion taking too long"

**Symptoms**: Running for 20+ hours, 0% GPU usage

**Solutions:**
1. Verify GPU is being used: Check logs for "using GPU: cuda"
2. Restart if GPU not detected (should auto-detect)
3. Use environment variable to select specific GPU: `$env:CUDA_VISIBLE_DEVICES="0"`
4. Consider faster parser if GPU unavailable (`spacy` instead of `stanza`)

### Issue: "Out of memory during ingestion"

**Solutions:**
1. Use `source` (JSONL) instead of `input_dir` for streaming
2. Reduce `chunk_size` (try 256 instead of 384)
3. Disable `relation_model` temporarily
4. Process in batches (split JSONL into multiple files)

## Quality Metrics

After ingestion, you should see:

**Good Dataset Indicators:**
- 📊 **Chunks per document**: 8-12 avg (for 384 chunk_size)
- 📊 **Relations per chunk**: 2-5 avg
- 📊 **Total relations**: Should be 2-5x the number of chunks
- 📊 **Unique relation types**: 20-50 different predicates

**Example for 64K documents:**
```
Documents: 64,000
Chunks: ~640,000 (10 per doc avg)
Relations: ~2,560,000 (4 per chunk avg)
Processing time: 3-5 hours with GPU
```

**If your metrics are significantly lower**, review document quality and parser configuration.

## Next Steps

After successful ingestion:

1. **Build training datasets** with optimal confidence filtering:
   ```bash
   python -m sphana_trainer.cli dataset-build-from-ingest target/ingest/ \
       --output-dir target/datasets/ \
       --min-confidence 0.35 \
       --val-ratio 0.15 \
       --seed 42
   ```

2. **Train models** in sequence:
   ```bash
   python -m sphana_trainer.cli train embedding --config configs/embedding/wiki.yaml
   python -m sphana_trainer.cli train relation --config configs/relation/wiki.yaml
   python -m sphana_trainer.cli train gnn --config configs/gnn/wiki.yaml
   ```

See [Complete Training Workflow](./trainer-workflow) for the full pipeline.

## Additional Resources

- [CLI Reference](./trainer-cli) - All ingestion commands
- [Configuration Guide](./trainer-config) - YAML configuration details
- [Training Workflow](./trainer-workflow) - End-to-end pipeline
- Sample data: `services/Sphana.Trainer/samples/wiki-docs/`
