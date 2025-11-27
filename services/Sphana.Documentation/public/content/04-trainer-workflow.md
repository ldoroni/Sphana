---
title: Training Workflow
description: End-to-end model training pipeline
---

# Training Workflow

This guide walks you through the complete model training workflow from raw documents to deployed ONNX models.

## Complete Training Pipeline

The recommended workflow follows these stages:

### 1. Download Training Data

```bash
python -m sphana_trainer.cli dataset-download-wiki \
    --titles-dir samples/wiki-titles/large/ \
    --full-content \
    --output samples/wiki-docs/large/docs.jsonl \
    --limit 500000
```

This downloads Wikipedia articles based on titles in `samples/wiki-titles/`. The output is a JSONL file where each line contains:

```json
{
  "title": "Neural Network",
  "content": "A neural network is...",
  "url": "https://en.wikipedia.org/wiki/Neural_network"
}
```

### 2. Ingest and Preprocess

Process raw documents into chunks and extract relations:

```bash
python -m sphana_trainer.cli ingest --config configs/ingest/wiki.yaml
```

**Output**: `target/ingest/`
- `chunks.jsonl` - Document chunks with metadata
- `relations.jsonl` - Entity-relation triples
- `cache/` - Cached parse trees

### 3. Build Training Datasets

Convert ingestion output into model-specific training datasets:

```bash
python -m sphana_trainer.cli dataset-build-from-ingest target/ingest/ \
    --output-dir target/datasets/ \
    --min-confidence 0.3 \
    --val-ratio 0.2 \
    --seed 42
```

**Output**: `target/datasets/`
- `embedding/train.jsonl` & `embedding/val.jsonl`
- `relation/train.jsonl` & `relation/val.jsonl`
- `gnn/train.jsonl` & `gnn/val.jsonl`

### 4. Train Models

Train each component sequentially:

```bash
# Embedding model (~1-2 hours on GPU)
python -m sphana_trainer.cli train embedding --config configs/embedding/wiki.yaml

# Relation extraction model (~2-3 hours on GPU)
python -m sphana_trainer.cli train relation --config configs/relation/wiki.yaml

# GNN ranker (~3-4 hours on GPU)
python -m sphana_trainer.cli train gnn --config configs/gnn/wiki.yaml

# NER model (~1-2 hours on GPU)
python -m sphana_trainer.cli train ner --config configs/ner/base.yaml

# LLM generator (~4-6 hours on GPU)
python -m sphana_trainer.cli train llm --config configs/llm/base.yaml
```

Each training run:
1. Loads dataset
2. Trains model with checkpointing
3. Validates on validation set
4. Exports PyTorch checkpoint
5. Logs metrics to `target/logs/` and optionally MLflow

### 5. Export to ONNX

Convert trained models to ONNX format with quantization:

```bash
python -m sphana_trainer.cli export --config configs/export/wiki.yaml
```

This command:
- Loads latest checkpoints for each component
- Converts to ONNX with dynamic axes
- Applies INT8 quantization (~50% size reduction)
- Runs parity checks (PyTorch vs ONNX < 1% diff)
- Generates manifest at `target/manifests/latest.json`

### 6. Package for Deployment

Bundle everything into a tarball:

```bash
python -m sphana_trainer.cli package --config configs/export/wiki.yaml
```

Creates `target/manifests/latest.tar.gz` containing:
- All ONNX model files
- Tokenizers and vocabularies
- Model manifest (versions, metrics, hashes)

## Single-Command Workflow

Run the entire pipeline with one command:

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

## Workflow Stages

### Stage 1: Data Preparation

**Input**: Raw text documents  
**Output**: Ingested chunks and relations  
**Duration**: 1-2 hours for 500k documents

Key considerations:
- Choose appropriate `parser` backend (simple/spacy/stanza)
- Balance `chunk_size` vs context quality
- Set `min_confidence` to filter low-quality relations

### Stage 2: Dataset Construction

**Input**: Ingestion output  
**Output**: Training-ready JSONL files  
**Duration**: 10-30 minutes

Key considerations:
- Set appropriate `val_ratio` (typically 0.1-0.2)
- Use fixed `seed` for reproducibility
- Add `--extra-*` datasets for domain adaptation

### Stage 3: Model Training

**Input**: Training datasets  
**Output**: PyTorch checkpoints  
**Duration**: 4-12 hours total

Training order matters:
1. **Embedding** first - needed by other models
2. **Relation** second - uses embeddings
3. **GNN** third - uses relation data
4. **NER** optional - can run in parallel
5. **LLM** last - uses full pipeline

### Stage 4: ONNX Export

**Input**: PyTorch checkpoints  
**Output**: Quantized ONNX models  
**Duration**: 10-20 minutes

Key checks:
- Parity validation (< 1% difference)
- Model size reduction (should be ~50% with INT8)
- Dynamic axes support for batching

### Stage 5: Packaging

**Input**: ONNX models  
**Output**: Deployment tarball  
**Duration**: 1-2 minutes

Manifest includes:
- Model versions and hashes
- Performance metrics
- Compatible CUDA/ORT versions

## Distributed Training

For multi-GPU training:

```bash
torchrun --nproc_per_node=2 \
    -m sphana_trainer.cli train embedding \
    --config configs/embedding/base.yaml \
    --resume latest
```

Requirements:
- Set `ddp: true` in config
- Use `precision: fp16` or `bf16`
- Batch size scales automatically

## Monitoring Training

### Real-time Logs

```bash
tail -f target/logs/embedding-train-*.log
```

### MLflow UI

```bash
mlflow ui --backend-store-uri target/mlruns
# Open http://localhost:5000
```

### Metrics JSON

```bash
cat target/artifacts/embedding/latest/metrics.jsonl
```

Each line contains:
```json
{
  "epoch": 1,
  "train_loss": 0.234,
  "val_cosine_sim": 0.872,
  "throughput_samples_per_sec": 1234.5
}
```

## Quick Testing Workflow

For rapid iteration during development:

```bash
# Use small test datasets
python -m sphana_trainer.cli train embedding --config configs/embedding/base.yaml

# Skip quantization for faster export
python -m sphana_trainer.cli export --config configs/export/base.yaml --skip-quantization

# Test exported models immediately
python -m sphana_trainer.cli artifacts parity-samples embedding 0.1.0
```

## Troubleshooting

### Training is slow

- Reduce batch size if GPU memory is low
- Use mixed precision (`precision: fp16`)
- Enable gradient accumulation
- Check GPU utilization with `nvidia-smi`

### Out of memory

- Reduce `batch_size`
- Reduce `max_seq_length`
- Use gradient checkpointing
- Train on CPU (much slower)

### Quantization fails

- Some models don't support INT8 quantization
- Try `--skip-quantization` and use FP32
- Check ONNX Runtime compatibility

### Validation metrics are poor

- Increase training `epochs`
- Try different learning rates
- Check dataset quality
- Ensure train/val split is representative

## Next Steps

- Learn about [Configuration](/trainer-config) options in detail
- Explore [Advanced Topics](/advanced) for optimization techniques
- See [Integration Guide](/integration) for deploying models

