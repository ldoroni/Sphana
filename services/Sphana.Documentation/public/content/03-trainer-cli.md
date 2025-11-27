---
title: Trainer CLI Reference
description: Complete command reference for the Sphana Trainer CLI
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

## Dataset Commands

### dataset-download-wiki

Download Wikipedia articles for training:

```bash
python -m sphana_trainer.cli dataset-download-wiki \
    --titles-dir samples/wiki-titles/large/ \
    --full-content \
    --output samples/wiki-docs/large/docs.jsonl \
    --limit 500000
```

**Options:**
- `--titles-dir`: Directory containing wiki title files
- `--titles-file`: Single file with wiki titles
- `--full-content`: Download full article content (not just intro)
- `--output`: Output JSONL file path
- `--limit`: Maximum number of articles to download

### dataset-build-from-ingest

Convert ingestion output to training datasets:

```bash
python -m sphana_trainer.cli dataset-build-from-ingest target/ingest/ \
    --output-dir target/datasets/ \
    --min-confidence 0.3 \
    --val-ratio 0.2 \
    --seed 42
```

**Options:**
- `--output-dir`: Directory for training datasets
- `--min-confidence`: Minimum confidence score for relations
- `--val-ratio`: Validation set ratio (0.0-1.0)
- `--seed`: Random seed for reproducibility

### dataset-validate

Validate dataset format:

```bash
python -m sphana_trainer.cli dataset-validate \
    src/tests/data/embedding/train.jsonl \
    --type embedding
```

**Options:**
- `--type`: Dataset type (`embedding`, `relation`, `gnn`)

### dataset-stats

Show dataset statistics:

```bash
python -m sphana_trainer.cli dataset-stats src/tests/data/embedding/train.jsonl
```

## Ingestion Commands

### ingest

Process raw documents and prepare training data:

```bash
python -m sphana_trainer.cli ingest --config configs/ingest/wiki.yaml
```

**Configuration** (`configs/ingest/wiki.yaml`):
```yaml
ingest:
  input_dir: "samples/wiki-docs/large"
  artifact_root: "target/ingest"
  parser: "simple"  # or "spacy", "stanza"
  chunk_size: 512
  chunk_overlap: 50
  relation_model: null  # Optional: HF model for RE
```

Outputs:
- `chunks.jsonl` - Document chunks with embeddings
- `relations.jsonl` - Extracted entity-relation triples
- `cache/` - Cached parse trees and chunks

### ingest-validate

Validate ingestion output:

```bash
python -m sphana_trainer.cli ingest-validate \
    --config configs/ingest/base.yaml \
    --stats \
    --chunks-schema src/sphana_trainer/schemas/ingestion/chunks.schema.json \
    --relations-schema src/sphana_trainer/schemas/ingestion/relations.schema.json
```

## Training Commands

### train embedding

Train the embedding model:

```bash
python -m sphana_trainer.cli train embedding --config configs/embedding/wiki.yaml
```

**Configuration** (`configs/embedding/wiki.yaml`):
```yaml
embedding:
  base_model: "sentence-transformers/all-MiniLM-L6-v2"
  output_dimension: 384
  max_seq_length: 512
  
training:
  batch_size: 32
  epochs: 3
  learning_rate: 2e-5
  warmup_steps: 100
  
dataset:
  train_file: "target/datasets/embedding/train.jsonl"
  validation_file: "target/datasets/embedding/val.jsonl"
```

### train relation

Train the relation extraction model:

```bash
python -m sphana_trainer.cli train relation --config configs/relation/wiki.yaml
```

**Configuration** (`configs/relation/wiki.yaml`):
```yaml
relation:
  base_model: "bert-base-uncased"
  num_labels: 42  # TACRED has 42 relation types
  
training:
  batch_size: 16
  epochs: 5
  learning_rate: 3e-5
  
dataset:
  train_file: "target/datasets/relation/train.jsonl"
  validation_file: "target/datasets/relation/val.jsonl"
```

### train gnn

Train the GNN ranker model:

```bash
python -m sphana_trainer.cli train gnn --config configs/gnn/wiki.yaml
```

**Configuration** (`configs/gnn/wiki.yaml`):
```yaml
gnn:
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  
training:
  batch_size: 8
  epochs: 10
  learning_rate: 1e-3
  
dataset:
  train_file: "target/datasets/gnn/train.jsonl"
  validation_file: "target/datasets/gnn/val.jsonl"
```

### train ner

Train the Named Entity Recognition model:

```bash
python -m sphana_trainer.cli train ner --config configs/ner/base.yaml
```

### train llm

Train or fine-tune the LLM generator:

```bash
python -m sphana_trainer.cli train llm --config configs/llm/base.yaml
```

**Configuration** (`configs/llm/base.yaml`):
```yaml
llm:
  model_name: "Qwen/Qwen2-0.5B-Instruct"
  # or: "meta-llama/Llama-3.2-1B"
  # or: "google/gemma-2-2b-it"
  quantize: true
  max_seq_length: 2048
  
training:
  epochs: 1
  batch_size: 2
  learning_rate: 1e-5
```

### train sweep

Run hyperparameter sweep:

```bash
python -m sphana_trainer.cli train sweep embedding \
    --config configs/embedding/base.yaml \
    --lr 2e-5 --lr 5e-5 \
    --batch-size 16 --batch-size 32 \
    --temperature 0.05 --temperature 0.07
```

## Export & Package Commands

### export

Export trained models to ONNX:

```bash
python -m sphana_trainer.cli export --config configs/export/wiki.yaml
```

This command:
1. Validates all required models are trained
2. Exports to ONNX with INT8 quantization
3. Runs parity checks (PyTorch vs ONNX)
4. Generates `target/manifests/latest.json`

### package

Package ONNX models into a tarball:

```bash
python -m sphana_trainer.cli package --config configs/export/wiki.yaml
```

Creates `target/manifests/latest.tar.gz` containing:
- ONNX model files
- Manifest JSON
- Tokenizers and vocabularies

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

## Workflow Commands

### workflow run

Run the complete training workflow:

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

This command orchestrates:
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

Check workflow progress:

```bash
python -m sphana_trainer.cli workflow status --artifact-root target/artifacts
```

## Configuration Options

All training commands support these common options:

- `--config`: Path to YAML configuration file
- `--resume latest`: Resume from latest checkpoint
- `--device cuda`: Force specific device (cuda/cpu)
- `--seed 42`: Set random seed
- `--log-to-mlflow`: Enable MLflow logging

## Output Locations

- **Artifacts**: `target/artifacts/<component>/<version>/`
- **Datasets**: `target/datasets/<component>/`
- **Logs**: `target/logs/`
- **MLflow**: `target/mlruns/`
- **Manifests**: `target/manifests/`

## Environment Variables

- `PYTHONPATH`: Set to `"src"` for module imports
- `SPHANA_WORKSPACE`: Override output location
- `CUDA_VISIBLE_DEVICES`: Select specific GPU

## Tips

- Use `--help` with any command for detailed options
- Start with small datasets for testing (`samples/wiki-titles/small/`)
- Monitor GPU usage with `nvidia-smi`
- Check `target/logs/` for detailed training logs
- Use MLflow UI to visualize training: `mlflow ui --backend-store-uri target/mlruns`

