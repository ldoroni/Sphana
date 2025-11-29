---
title: Configuration
description: YAML configuration files explained
---

# Configuration

Sphana Trainer uses YAML configuration files to manage all training parameters. This page explains the structure and options for each component.

## Configuration File Structure

All config files follow a similar structure:

```yaml
# Component-specific settings
<component>:
  # Model architecture
  base_model: "model-name"
  output_dimension: 384
  
  # ... component-specific options

# Training hyperparameters
training:
  batch_size: 32
  epochs: 3
  learning_rate: 2e-5
  warmup_steps: 100
  precision: "fp32"  # or "fp16", "bf16"
  
# Dataset configuration
dataset:
  train_file: "path/to/train.jsonl"
  validation_file: "path/to/val.jsonl"
  
# Export settings
export:
  quantize: true
  validate_parity: true
  
# Optional: MLflow logging
mlflow:
  log_to_mlflow: true
  tracking_uri: "target/mlruns"
  experiment_name: "sphana-training"
```

## Embedding Configuration

**File**: `configs/embedding/base.yaml` or `configs/embedding/wiki.yaml`

```yaml
embedding:
  base_model: "sentence-transformers/all-MiniLM-L6-v2"
  output_dimension: 384
  max_seq_length: 512
  pooling_mode: "mean"  # mean, cls, max
  normalize_embeddings: true

training:
  batch_size: 32
  epochs: 3
  learning_rate: 2e-5
  warmup_steps: 100
  weight_decay: 0.01
  temperature: 0.05  # Contrastive loss temperature
  precision: "fp32"
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0

dataset:
  train_file: "target/datasets/embedding/train.jsonl"
  validation_file: "target/datasets/embedding/val.jsonl"

export:
  version: "0.1.0"
  artifact_root: "target/artifacts"
  quantize: true
  validate_parity: true
  metric_threshold:
    val_cosine_sim: 0.80  # Minimum validation cosine similarity

mlflow:
  log_to_mlflow: false
  tracking_uri: null
  experiment_name: "embedding-training"
```

**Key Options**:
- `temperature`: Lower values make contrastive learning more discriminative
- `normalize_embeddings`: Required for cosine similarity search
- `metric_threshold`: Block export if validation metrics are poor

## Relation Extraction Configuration

**File**: `configs/relation/base.yaml` or `configs/relation/wiki.yaml`

```yaml
relation:
  base_model: "bert-base-uncased"
  num_labels: 42  # TACRED: 42 relation types (including no_relation)
  max_seq_length: 128
  entity_markers: true  # Use [E1], [/E1], [E2], [/E2] markers
  
training:
  batch_size: 16
  epochs: 5
  learning_rate: 3e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
  precision: "fp32"
  label_smoothing: 0.1  # Helps with class imbalance

dataset:
  train_file: "target/datasets/relation/train.jsonl"
  validation_file: "target/datasets/relation/val.jsonl"

export:
  version: "0.1.0"
  artifact_root: "target/artifacts"
  quantize: true
  metric_threshold:
    val_macro_f1: 0.70  # Minimum F1 score

mlflow:
  log_to_mlflow: false
```

**Dataset Format**:
```json
{
  "text": "John works at Google in Mountain View.",
  "entity1": {"text": "John", "start": 0, "end": 4},
  "entity2": {"text": "Google", "start": 14, "end": 20},
  "label": "per:employee_of"
}
```

## GNN Configuration

**File**: `configs/gnn/base.yaml` or `configs/gnn/wiki.yaml`

```yaml
gnn:
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  bidirectional: true  # Use bi-directional GGNN
  aggregation: "max"  # max, mean, attention

training:
  batch_size: 8
  epochs: 10
  learning_rate: 1e-3
  warmup_steps: 50
  weight_decay: 1e-4
  listwise_loss: "listnet"  # or "listmle"
  precision: "fp16"  # GNN training benefits from fp16

dataset:
  train_file: "target/datasets/gnn/train.jsonl"
  validation_file: "target/datasets/gnn/val.jsonl"

export:
  version: "0.1.0"
  artifact_root: "target/artifacts"
  quantize: true
  dynamic_axes: true  # Required for variable graph sizes
```

**Dataset Format**:
```json
{
  "query_id": "q1",
  "candidates": [
    {
      "node_features": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
      "edge_index": [[0, 1], [1, 2]],
      "edge_directions": [0, 1],
      "label": 1.0
    },
    {
      "node_features": [...],
      "edge_index": [...],
      "edge_directions": [...],
      "label": 0.5
    }
  ]
}
```

## LLM Configuration

**File**: `configs/llm/base.yaml`

```yaml
llm:
  model_name: "Qwen/Qwen2-0.5B-Instruct"
  # Alternatives:
  # "meta-llama/Llama-3.2-1B"
  # "google/gemma-2-2b-it"
  
  max_seq_length: 2048
  quantize: true
  use_flash_attention: false  # Enable if supported
  
training:
  epochs: 1
  batch_size: 2
  learning_rate: 1e-5
  warmup_steps: 100
  gradient_accumulation_steps: 4
  precision: "bf16"  # BFloat16 recommended for LLMs
  
dataset:
  train_file: null  # Optional: fine-tuning dataset
  validation_file: null

export:
  version: "0.1.0"
  artifact_root: "target/artifacts"
  quantize: true
  external_data: true  # Store weights separately
```

> **Note**: LLM training requires authentication with Hugging Face for Llama models. Run `huggingface-cli login` first.

## Ingestion Configuration

**File**: `configs/ingest/base.yaml` or `configs/ingest/wiki.yaml`

```yaml
ingest:
  input_dir: "samples/wiki-docs/large"
  artifact_root: "target/ingest"
  
  # Parser backend: simple, spacy, stanza
  parser: "simple"
  
  # Chunking parameters
  chunk_size: 512
  chunk_overlap: 50
  min_chunk_length: 50
  
  # Relation extraction
  relation_model: null  # Optional: "hf-model-name"
  min_relation_confidence: 0.5
  
  # Caching
  cache_chunks: true
  cache_parses: true
  
  # Parallel processing
  max_workers: 4
  
  # Progress logging
  progress_log_interval: 10  # Log every 10%
```

**Parser Options**:
- `simple`: Basic regex-based parsing (fast)
- `spacy`: spaCy dependency parsing (requires `en_core_web_sm`)
- `stanza`: Stanford NLP parsing (most accurate)

## Export Configuration

**File**: `configs/export/base.yaml` or `configs/export/wiki.yaml`

```yaml
export:
  components:
    - embedding
    - relation
    - gnn
    - ner
    - llm
  
  artifact_root: "target/artifacts"
  manifest_path: "target/manifests/latest.json"
  
  # Quantization settings
  quantize: true
  quantization_mode: "int8"  # int8 or int16
  calibration_samples: 100
  
  # Validation
  validate_parity: true
  max_parity_diff: 0.01  # 1% maximum difference
  
  # Model selection
  use_latest: true  # Use latest checkpoints
  versions:
    embedding: null  # or specific version like "0.1.0"
    relation: null
    gnn: null
```

## Progress Logging Configuration

### progress_log_interval

**Type**: Integer (1-100)  
**Default**: 10  
**Available in**: All configs (ingest, embedding, relation, gnn, ner, llm)

Controls how frequently progress updates are logged during long-running operations.

**Value meanings:**
- `1`: Log every 1% (100 total logs) - Very detailed
- `5`: Log every 5% (20 total logs) - Detailed
- `10`: Log every 10% (10 total logs) - Moderate (default)
- `25`: Log every 25% (4 total logs) - Sparse

**Configuration:**
```yaml
ingest:
  progress_log_interval: 1  # For large datasets
  
embedding:
  progress_log_interval: 5  # For training visibility
```

**When to use each value:**

| Dataset Size | Operation Time | Recommended Interval | Rationale |
|--------------|----------------|----------------------|-----------|
| Small (<1K items) | < 10 minutes | 10-25 | Avoid log spam |
| Medium (1K-10K) | 10-60 minutes | 5-10 | Balanced visibility |
| Large (10K-100K) | 1-10 hours | 1-5 | Frequent reassurance |
| Very Large (100K+) | 10+ hours | 1 | Critical progress tracking |

**Example output with `progress_log_interval: 1`:**
```
Stage 1/1: Processing documents | 1% complete | 640/64000 items | Elapsed: 2m 10s | ETA: 3h 28m | Speed: 4.9 items/sec
Stage 1/1: Processing documents | 2% complete | 1280/64000 items | Elapsed: 4m 22s | ETA: 3h 30m | Speed: 4.9 items/sec
Stage 1/1: Processing documents | 3% complete | 1920/64000 items | Elapsed: 6m 35s | ETA: 3h 32m | Speed: 4.9 items/sec
```

**Progress log details:**
- **Stage**: Current processing stage (e.g., "Processing documents", "Training epoch 2")
- **% complete**: Percentage of current stage completed
- **Items**: Processed/Total items (docs, batches, etc.)
- **Elapsed**: Time since stage started
- **ETA**: Estimated time to completion based on current speed
- **Speed**: Processing throughput (items per second)

**Benefits:**
- Real-time progress feedback
- Accurate ETA calculations
- Performance monitoring (items/sec)
- Early problem detection (if speed drops significantly)

**Recommended values by command:**

| Command | Recommended | Why |
|---------|-------------|-----|
| `ingest` (large dataset) | 1 | Long operation (3-10 hours), need frequent updates |
| `train embedding` | 5 | Moderate epochs (20-40 min each), balanced visibility |
| `train relation` | 5 | Similar to embedding |
| `train gnn` | 5 | Quick convergence, moderate logging |
| `dataset-build-from-ingest` | 10 | Fast operation (5-15 min), default is fine |

---

## Common Training Options

All training configs support these common options:

### Optimization

```yaml
training:
  optimizer: "adamw"  # adamw, adam, sgd
  learning_rate: 2e-5
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8
  max_grad_norm: 1.0
```

### Learning Rate Scheduling

```yaml
training:
  scheduler: "linear"  # linear, cosine, constant
  warmup_steps: 100
  warmup_ratio: 0.1  # Alternative to warmup_steps
```

### Mixed Precision

```yaml
training:
  precision: "fp16"  # fp32, fp16, bf16
  # fp16: Faster, less memory, may have precision issues
  # bf16: Better for large models, requires Ampere GPU
  # fp32: Highest precision, slowest
```

### Checkpointing

```yaml
training:
  save_strategy: "epoch"  # epoch, steps
  save_steps: 1000
  save_total_limit: 3  # Keep only last 3 checkpoints
  load_best_model_at_end: true
```

### Early Stopping

```yaml
training:
  early_stopping_patience: 3  # Stop after 3 epochs without improvement
  early_stopping_threshold: 0.001  # Minimum improvement
```

### Distributed Training

```yaml
training:
  ddp: true  # Enable DistributedDataParallel
  # Launch with: torchrun --nproc_per_node=N
```

## MLflow Integration

Enable experiment tracking:

```yaml
mlflow:
  log_to_mlflow: true
  tracking_uri: "target/mlruns"  # Local or remote URI
  experiment_name: "sphana-embedding"
  run_name: "wiki-500k-v1"  # Optional custom name
  log_models: true  # Log model artifacts
  log_artifacts: true  # Log additional files
```

View experiments:

```bash
mlflow ui --backend-store-uri target/mlruns
# Open http://localhost:5000
```

## Configuration Overrides

Override config values via command line:

```bash
python -m sphana_trainer.cli train embedding \
    --config configs/embedding/base.yaml \
    --batch-size 64 \
    --learning-rate 3e-5 \
    --epochs 5
```

## Configuration Best Practices

### For Development

```yaml
training:
  batch_size: 8  # Small for fast iteration
  epochs: 1
  save_steps: 100
  gradient_accumulation_steps: 4  # Simulate larger batch

dataset:
  # Use small test datasets
  train_file: "src/tests/data/embedding/train.jsonl"
```

### For Production

```yaml
training:
  batch_size: 32  # Larger for stability
  epochs: 3-5
  precision: "fp16"  # Faster training
  ddp: true  # Multi-GPU

export:
  quantize: true  # Always quantize for production
  validate_parity: true  # Ensure quality
```

## Example Configurations

See the `configs/` directory for complete examples:

- `configs/embedding/base.yaml` - Minimal configuration
- `configs/embedding/wiki.yaml` - Production configuration for Wikipedia
- `configs/relation/wiki.yaml` - High-quality relation extraction
- `configs/gnn/wiki.yaml` - Graph neural network ranking

## Next Steps

- Explore the complete [Training Workflow](/trainer-workflow)
- Learn about [Advanced Topics](/advanced) like distributed training
- Check the [CLI Reference](/trainer-cli) for all available commands

