---
title: Trainer CLI Reference
description: Complete command reference for the Sphana Trainer CLI
category: Sphana Trainer
---

# Trainer CLI Reference

The Sphana Trainer CLI (`sphana_trainer`) provides commands for training, exporting, and managing neural models for the Sphana database.

## Installation

```bash
cd services/Sphana.Trainer
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set the PYTHONPATH before running any commands:

```bash
# Windows PowerShell
$env:PYTHONPATH="src"

# Linux/Mac
export PYTHONPATH="src"
```

## Core Training Workflow

The Sphana Trainer follows a sequential workflow. Each command depends on the output of the previous one:

```diagram
flowchart TD
    A[Download Data] --> B[Cache Models]
    B --> C[Ingest Documents]
    C --> D[Build Datasets]
    D --> E[Train Embedding]
    D --> F[Train Relation]
    D --> G[Train GNN]
    E --> H[Export ONNX]
    F --> H
    G --> H
    H --> I[Package Models]
```

---

## Detailed Command Reference

### 1. dataset-download-wiki

**Purpose**: Download Wikipedia articles to create a high-quality training corpus.

**What it does:**
- Fetches articles from Wikipedia API based on title lists
- Downloads full content or intro paragraphs only
- Saves to JSONL format for ingestion
- Handles rate limiting and retries automatically

**Input:**
- Title files (`.txt`) with one Wikipedia title per line
- OR single title file with all titles

**Output:**
- `docs.jsonl`: JSONL file with `id`, `title`, `text` fields
- Example: 100,000 articles ≈ 2-5 GB

**Why needed**: Provides diverse, high-quality, factual content for training. Wikipedia's encyclopedic style and rich entity relationships make it ideal for knowledge graph systems.

**Command:**
```bash
python -m sphana_trainer.cli dataset-download-wiki \
    --titles-dir samples/wiki-titles/large/ \
    --full-content \
    --output samples/wiki-docs/large/docs.jsonl \
    --limit 500000
```

**Parameters:**

| Parameter | Required | Description | Recommended Value |
|-----------|----------|-------------|-------------------|
| `--titles-dir` | One of titles-dir/titles-file | Directory with `.txt` title files | `samples/wiki-titles/large/` |
| `--titles-file` | One of titles-dir/titles-file | Single file with all titles | - |
| `--full-content` | No | Download complete articles vs intro only | ✅ Use for accuracy |
| `--output` | Yes | Output JSONL file path | `samples/wiki-docs/large/docs.jsonl` |
| `--limit` | No | Max articles to download | `500000` for large dataset |

**Typical duration**: 10-30 minutes for 10K articles, 2-6 hours for 100K articles (depends on network speed)

---

### 2. ingest-cache-models

**Purpose**: Pre-download NLP models before ingestion to avoid runtime downloads.

**What it does:**
- Downloads Stanza language models (~250 MB per language)
- Downloads spaCy models (13 MB - 587 MB depending on model)
- Caches models locally in `~/.stanza` or spaCy data directory
- One-time setup per model

**Input:** Language code or model name

**Output:** Cached models in home directory

**Why needed**: Separates model download from data processing. Prevents download failures during long-running ingestion jobs.

**Commands:**
```bash
# For Stanza parser (recommended for accuracy)
python -m sphana_trainer.cli ingest-cache-models --stanza-lang en

# For spaCy parser (faster alternative)
python -m sphana_trainer.cli ingest-cache-models --spacy-model en_core_web_trf
```

**Parameters:**

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `--stanza-lang` | Stanza language code | `en`, `fr`, `de`, `es`, `zh` |
| `--spacy-model` | spaCy model name | `en_core_web_sm`, `en_core_web_md`, `en_core_web_trf` |

**Typical duration**: 1-5 minutes per model

---

### 3. ingest

**Purpose**: Transform raw documents into structured training data (chunks + relations).

**What it does:**
1. Loads documents from JSONL file or directory
2. Normalizes text (removes extra whitespace, fixes encoding)
3. Splits documents into fixed-size chunks with overlap
4. Extracts entity-relation triples using NLP parser
5. Optionally classifies relations using HuggingFace model
6. Caches results for faster re-runs
7. Outputs `chunks.jsonl` and `relations.jsonl`

**Input:**
- Raw documents (JSONL file or text directory)
- Configuration file specifying chunking and parsing settings

**Output:**
- `target/ingest/chunks.jsonl`: Document chunks (for embedding training)
- `target/ingest/relations.jsonl`: Entity-relation triples (for relation/GNN training)
- `target/cache/`: Cached parses and intermediate results

**Why needed**: Creates the foundation for all downstream training. The quality of chunks and relations directly determines the accuracy of your trained models.

**Command:**
```bash
python -m sphana_trainer.cli ingest --config configs/ingest/wiki.yaml
```

**Key Configuration Parameters:**

| Parameter | Recommended | Range | Why It Matters |
|-----------|-------------|-------|----------------|
| `chunk_size` | **384** | 128-512 | Larger = better context, but slower training |
| `chunk_overlap` | **48** (12-15% of chunk_size) | 16-64 | Preserves relations at boundaries |
| `parser` | **stanza** | simple/spacy/stanza | Stanza = 95-97% accuracy, best for production |
| `relation_threshold` | **0.20** | 0.10-0.40 | Lower = more relations (higher recall) |
| `relation_model` | **facebook/bart-large-mnli** | Any HF model | Best quality relation classification |
| `relation_max_length` | **512** | 256-1024 | Matches BART's optimal sequence length |
| `progress_log_interval` | **1** | 1-25 | 1% = detailed progress for large datasets |

**Typical duration:**
- 10K docs: 30-60 minutes (with GPU)
- 64K docs: 3-5 hours (with GPU)
- 100K docs: 5-8 hours (with GPU)

**GPU Acceleration**: Automatically uses GPU if available via `torch.cuda.is_available()`. Stanza and BART models run 5-8x faster on GPU.

---

### 4. dataset-build-from-ingest

**Purpose**: Convert ingestion output into training-ready datasets with train/val splits.

**What it does:**
1. Reads `chunks.jsonl` and `relations.jsonl` from ingestion
2. Filters relations by confidence threshold
3. Creates training datasets for each model type:
   - **Embedding**: Anchor-positive pairs from chunks
   - **Relation**: Classified relation triples
   - **GNN**: Graph structures with node features and edges
4. Splits into training and validation sets
5. Balances classes and handles negative sampling

**Input:**
- Ingestion output directory (`target/ingest/`)
- Confidence threshold for filtering
- Validation ratio for train/val split

**Output:**
- `target/datasets/embedding/train.jsonl` + `validation.jsonl`
- `target/datasets/relation/train.jsonl` + `validation.jsonl`
- `target/datasets/gnn/train.jsonl` + `validation.jsonl`

**Why needed**: Raw ingestion data needs to be structured into specific formats for each model type. This command creates the specialized datasets with proper train/validation splits.

**Command:**
```bash
python -m sphana_trainer.cli dataset-build-from-ingest target/ingest/ \
    --output-dir target/datasets/ \
    --min-confidence 0.35 \
    --val-ratio 0.15 \
    --seed 42
```

**Parameters:**

| Parameter | Recommended | Range | Why It Matters |
|-----------|-------------|-------|----------------|
| `--min-confidence` | **0.35** | 0.20-0.50 | Filters low-quality relations while preserving recall |
| `--val-ratio` | **0.15** | 0.10-0.20 | 15% validation provides good evaluation without reducing training data |
| `--seed` | **42** | Any int | Ensures reproducible train/val splits |

**Typical duration**: 5-15 minutes for 64K documents

---

### 5. train embedding

**Purpose**: Train the sentence embedding model for semantic search.

**What it does:**
1. Loads embedding dataset (anchor-positive pairs)
2. Fine-tunes a transformer model (e.g., MiniLM) using contrastive learning
3. Trains model to produce similar embeddings for related chunks
4. Validates on held-out pairs
5. Exports to ONNX format with INT8 quantization
6. Saves checkpoint for further fine-tuning

**Input:**
- `target/datasets/embedding/train.jsonl`: Training pairs
- `target/datasets/embedding/validation.jsonl`: Validation pairs
- Base model from HuggingFace (e.g., `sentence-transformers/all-MiniLM-L6-v2`)

**Output:**
- `target/artifacts/embedding/<version>/checkpoint/`: PyTorch checkpoint
- `target/artifacts/embedding/<version>/onnx/model.onnx`: ONNX model
- `target/artifacts/embedding/<version>/onnx/model.quant.onnx`: Quantized ONNX
- Training metrics logged to MLflow

**Why needed**: The embedding model converts text chunks into dense vectors for semantic search. This is the core of the retrieval system.

**Command:**
```bash
python -m sphana_trainer.cli train embedding --config configs/embedding/wiki.yaml
```

**Key Configuration:**

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| `model_name` | sentence-transformers/all-MiniLM-L6-v2 | Fast, 384-dim, good quality |
| `batch_size` | 32 | Good balance for GPU memory |
| `epochs` | 3 | Sufficient for convergence |
| `learning_rate` | 2e-5 | Standard for fine-tuning transformers |
| `max_seq_length` | 512 | Matches chunk size |
| `precision` | fp16 | 2x faster training with minimal quality loss |
| `progress_log_interval` | 5 | Log every 5% during training |

**Typical duration**: 20-40 minutes per epoch (64K dataset on GPU)

---

### 6. train relation

**Purpose**: Train the relation classification model to identify relation types.

**What it does:**
1. Loads relation dataset (subject-predicate-object triples)
2. Fine-tunes BERT-based classifier for relation types
3. Learns to classify relations (e.g., "is_a", "part_of", "located_in")
4. Generates calibration coefficients for confidence scores
5. Validates using macro F1 score
6. Exports to ONNX format

**Input:**
- `target/datasets/relation/train.jsonl`: Labeled relations
- `target/datasets/relation/validation.jsonl`: Validation relations

**Output:**
- `target/artifacts/relation/<version>/checkpoint/`: Model checkpoint
- `target/artifacts/relation/<version>/checkpoint/calibration.json`: Confidence calibration
- `target/artifacts/relation/<version>/onnx/model.onnx`: ONNX model

**Why needed**: Classifies extracted relations into semantic types, enabling structured knowledge graph queries.

**Command:**
```bash
python -m sphana_trainer.cli train relation --config configs/relation/wiki.yaml
```

**Key Configuration:**

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| `model_name` | hf-internal-testing/tiny-random-bert | Lightweight for testing (replace with bert-base-uncased for production) |
| `batch_size` | 16 | Balance between speed and memory |
| `epochs` | 3 | Early stopping prevents overfitting |
| `max_seq_length` | 256 | Sufficient for relation contexts |
| `precision` | fp16 | Faster training on GPU |
| `progress_log_interval` | 5 | Moderate progress updates |

**Typical duration**: 30-60 minutes (64K dataset on GPU)

---

### 7. train gnn

**Purpose**: Train the Graph Neural Network for knowledge graph reasoning.

**What it does:**
1. Loads graph dataset (node features + edge structures)
2. Trains GGNN (Gated Graph Neural Network) for ranking
3. Learns to traverse knowledge graph and rank candidate answers
4. Optimizes using listwise ranking loss (ListNet/ListMLE/ApproxNDCG)
5. Validates on query ranking tasks
6. Exports to ONNX format

**Input:**
- `target/datasets/gnn/train.jsonl`: Graph queries with candidates
- `target/datasets/gnn/validation.jsonl`: Validation queries

**Output:**
- `target/artifacts/gnn/<version>/checkpoint/`: Model checkpoint
- `target/artifacts/gnn/<version>/onnx/model.onnx`: ONNX model

**Why needed**: Enables multi-hop reasoning over the knowledge graph by learning to rank paths and entities.

**Command:**
```bash
python -m sphana_trainer.cli train gnn --config configs/gnn/wiki.yaml
```

**Key Configuration:**

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| `hidden_dim` | 64 | Sufficient for most graphs |
| `num_layers` | 3 | Captures 3-hop neighborhood |
| `batch_size` | 1 | Each query is a full graph |
| `epochs` | 2 | Quick convergence |
| `listwise_loss` | listnet | Differentiable listwise ranking |
| `precision` | fp16 | Faster training |
| `progress_log_interval` | 5 | Moderate updates |

**Typical duration**: 15-30 minutes (64K dataset on GPU)

---

### 8. train ner

**Purpose**: Train the Named Entity Recognition model.

**What it does:**
- Fine-tunes a sequence labeling model for entity detection
- Identifies entities (PERSON, ORG, LOC, etc.) in text
- Uses BIO tagging scheme

**Command:**
```bash
python -m sphana_trainer.cli train ner --config configs/ner/base.yaml
```

**Why needed**: Extracts entities for knowledge graph population and query understanding.

---

### 9. train llm

**Purpose**: Train or fine-tune the LLM generator for answer synthesis.

**What it does:**
- Fine-tunes a small language model (Qwen, Llama, Gemma)
- Trains on question-answer pairs
- Learns to synthesize natural language responses
- Supports quantization for efficient inference

**Command:**
```bash
python -m sphana_trainer.cli train llm --config configs/llm/base.yaml
```

**Configuration:**
```yaml
llm:
  model_name: "Qwen/Qwen2-0.5B-Instruct"
  # or: "meta-llama/Llama-3.2-1B"
  # or: "google/gemma-2-2b-it"
  quantize: true
  max_seq_length: 2048
  
  epochs: 1
  batch_size: 2
  learning_rate: 1e-5
```

**Why needed**: Generates fluent, context-aware answers from retrieved knowledge graph information.

---

### 10. export

**Purpose**: Export trained PyTorch models to ONNX format for deployment.

**What it does:**
1. Validates all required models are trained
2. Converts PyTorch models to ONNX format
3. Applies INT8 quantization for faster inference
4. Runs parity checks (PyTorch vs ONNX outputs)
5. Generates manifest JSON with model metadata

**Input:**
- Trained model checkpoints in `target/artifacts/`

**Output:**
- ONNX models in each artifact directory
- `target/manifests/latest.json`: Model manifest

**Why needed**: ONNX format enables cross-platform deployment and optimized inference in the .NET Sphana Database service.

**Command:**
```bash
python -m sphana_trainer.cli export --config configs/export/wiki.yaml
```

**Typical duration**: 5-10 minutes

---

### 11. package

**Purpose**: Package ONNX models into a tarball for deployment.

**What it does:**
- Bundles all ONNX models into a single archive
- Includes manifest, tokenizers, and vocabularies
- Creates versioned tarball for deployment

**Input:**
- ONNX models from export step
- Manifest JSON

**Output:**
- `target/manifests/latest.tar.gz`: Complete model package

**Why needed**: Simplifies deployment by bundling all required files into a single deployable artifact.

**Command:**
```bash
python -m sphana_trainer.cli package --config configs/export/wiki.yaml
```

---

## Validation Commands

### dataset-validate

Validate dataset format and schema compliance:

```bash
python -m sphana_trainer.cli dataset-validate \
    target/datasets/embedding/train.jsonl \
    --type embedding
```

**Options:**
- `--type`: Dataset type (`embedding`, `relation`, `gnn`)

### dataset-stats

Show dataset statistics and distributions:

```bash
python -m sphana_trainer.cli dataset-stats target/datasets/embedding/train.jsonl
```

### ingest-validate

Validate ingestion output:

```bash
python -m sphana_trainer.cli ingest-validate \
    --config configs/ingest/wiki.yaml \
    --stats \
    --chunks-schema src/sphana_trainer/schemas/ingestion/chunks.schema.json \
    --relations-schema src/sphana_trainer/schemas/ingestion/relations.schema.json
```

---

## Artifact Management

### artifacts list

List all trained artifacts:

```bash
python -m sphana_trainer.cli artifacts list --artifact-root target/artifacts
```

### artifacts promote

Promote a specific version to production:

```bash
python -m sphana_trainer.cli artifacts promote embedding 0.1.0 \
    --artifact-root target/artifacts \
    --manifest target/manifests/promoted.json
```

### artifacts bundle

Bundle artifacts for deployment:

```bash
python -m sphana_trainer.cli artifacts bundle embedding 0.1.0 target/bundles/embedding \
    --artifact-root target/artifacts
```

---

## Workflow Commands

### workflow run

Run the complete training workflow end-to-end:

```bash
python -m sphana_trainer.cli workflow run \
    --ingest-config configs/ingest/base.yaml \
    --embedding-config configs/embedding/base.yaml \
    --relation-config configs/relation/base.yaml \
    --gnn-config configs/gnn/base.yaml \
    --export-config configs/export/base.yaml \
    --package-config configs/export/base.yaml \
    --promote-component embedding \
    --promote-version 0.1.0 \
    --manifest target/manifests/latest.json \
    --build-datasets \
    --dataset-output-dir target/datasets/wiki
```

This orchestrates:
1. Data ingestion
2. Dataset building
3. Model training (all components)
4. ONNX export
5. Packaging
6. Artifact promotion

### workflow wiki

Run the Wikipedia-optimized workflow:

```bash
python -m sphana_trainer.cli workflow wiki --artifact-root target/artifacts
```

### workflow status

Check workflow progress and completion:

```bash
python -m sphana_trainer.cli workflow status --artifact-root target/artifacts
```

---

## Progress Logging Configuration

All commands now support configurable progress logging:

```yaml
# In any config file (ingest, embedding, relation, gnn)
progress_log_interval: 1  # Log every 1%

# Options:
# - 1 = 100 logs (very detailed, for large datasets)
# - 5 = 20 logs (detailed, for training)
# - 10 = 10 logs (moderate, default)
# - 25 = 4 logs (sparse, for quick operations)
```

**Recommendations:**
- **Ingestion (64K docs)**: `progress_log_interval: 1` (logs every 1% ≈ every 2 minutes)
- **Training (3 epochs)**: `progress_log_interval: 5` (logs every 5% ≈ every 1-2 minutes per epoch)
- **Quick operations**: `progress_log_interval: 10` (default)

**Example log output:**
```
Stage 1/1: Processing documents | 15% complete | 9600/64000 items | Elapsed: 28m 30s | ETA: 2h 42m | Speed: 5.6 items/sec
```

---

## Common Options

All training commands support these options:

- `--config`: Path to YAML configuration file (required)
- `--resume latest`: Resume from latest checkpoint
- `--seed 42`: Set random seed for reproducibility
- `--log-to-mlflow`: Enable MLflow experiment tracking

**Note**: Device selection (GPU/CPU) is now automatic via `torch.cuda.is_available()`. To force a specific GPU, use environment variable: `$env:CUDA_VISIBLE_DEVICES="0"`

---

## Output Locations

- **Artifacts**: `target/artifacts/<component>/<version>/`
- **Datasets**: `target/datasets/<component>/`
- **Ingestion**: `target/ingest/`
- **Cache**: `target/cache/`
- **Logs**: `target/logs/`
- **MLflow**: `target/mlruns/`
- **Manifests**: `target/manifests/`

---

## Environment Variables

- `PYTHONPATH`: Set to `"src"` for module imports (required)
- `SPHANA_WORKSPACE`: Override output location (optional)
- `CUDA_VISIBLE_DEVICES`: Select specific GPU (e.g., `"0"` for first GPU)

---

## Tips

- Use `--help` with any command for detailed options
- Start with small datasets for testing (`samples/wiki-titles/small/`)
- Monitor GPU usage with `nvidia-smi` or Task Manager
- Check `target/logs/` for detailed training logs
- Use MLflow UI to visualize training: `mlflow ui --backend-store-uri target/mlruns`
- Cache ingestion output for faster iterations
- Use progress logs to estimate completion time
