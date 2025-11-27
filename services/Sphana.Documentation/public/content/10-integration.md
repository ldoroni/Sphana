---
title: Integration Guide
description: Python to .NET integration workflow
---

# Integration Guide

This guide explains how trained models flow from the Python Trainer to the .NET Database service.

## Integration Overview

```
┌─────────────────────────────────┐
│   Python Trainer CLI            │
│                                 │
│  1. Train PyTorch models        │
│  2. Export to ONNX              │
│  3. Apply INT8 quantization     │
│  4. Generate manifest           │
│  5. Package artifacts           │
└────────────┬────────────────────┘
             │ (tarball)
             ▼
┌─────────────────────────────────┐
│   Model Artifact Registry       │
│                                 │
│  target/artifacts/              │
│  ├─ embedding/0.1.0/            │
│  ├─ relation/0.1.0/             │
│  ├─ gnn/0.1.0/                  │
│  └─ llm/0.1.0/                  │
│                                 │
│  target/manifests/latest.json   │
└────────────┬────────────────────┘
             │ (copy models)
             ▼
┌─────────────────────────────────┐
│   .NET Database Service         │
│                                 │
│  1. Load manifest               │
│  2. Load ONNX models            │
│  3. Initialize ONNX Runtime     │
│  4. Create session pools        │
│  5. Start gRPC service          │
└─────────────────────────────────┘
```

## Step-by-Step Integration

### Step 1: Train Models (Python)

Train all required components:

```bash
cd services/Sphana.Trainer
$env:PYTHONPATH="src"

# Train all models
python -m sphana_trainer.cli train embedding --config configs/embedding/wiki.yaml
python -m sphana_trainer.cli train relation --config configs/relation/wiki.yaml
python -m sphana_trainer.cli train gnn --config configs/gnn/wiki.yaml
python -m sphana_trainer.cli train llm --config configs/llm/base.yaml
```

Output: PyTorch checkpoints in `target/artifacts/<component>/latest/pytorch/`

### Step 2: Export to ONNX (Python)

Convert PyTorch models to ONNX format:

```bash
python -m sphana_trainer.cli export --config configs/export/wiki.yaml
```

This command:
1. Loads latest PyTorch checkpoints
2. Converts to ONNX with dynamic axes
3. Applies INT8 quantization (~50% size reduction)
4. Runs parity checks (PyTorch vs ONNX)
5. Generates `target/manifests/latest.json`

Output:
- ONNX files: `target/artifacts/<component>/0.1.0/onnx/*.onnx`
- Manifest: `target/manifests/latest.json`

### Step 3: Package Artifacts (Python)

Bundle everything for deployment:

```bash
python -m sphana_trainer.cli package --config configs/export/wiki.yaml
```

Output: `target/manifests/latest.tar.gz` containing:
- All ONNX model files
- Tokenizers and vocabularies  
- Model manifest (versions, metrics, checksums)

### Step 4: Extract Models (.NET side)

Copy the ONNX models to the .NET service:

```bash
# Option 1: Copy from artifacts directory
cd services/Sphana.Database
mkdir -p Sphana.Database/Resources/Models
cp ../Sphana.Trainer/target/artifacts/embedding/0.1.0/onnx/embedding.onnx Sphana.Database/Resources/Models/
cp ../Sphana.Trainer/target/artifacts/relation/0.1.0/onnx/relation_extraction.onnx Sphana.Database/Resources/Models/
cp ../Sphana.Trainer/target/artifacts/gnn/0.1.0/onnx/gnn_ranker.onnx Sphana.Database/Resources/Models/
cp ../Sphana.Trainer/target/artifacts/llm/0.1.0/onnx/llm_generator.onnx Sphana.Database/Resources/Models/

# Also copy tokenizers/vocabularies
cp ../Sphana.Trainer/target/artifacts/llm/0.1.0/onnx/vocab.json Sphana.Database/Resources/Vocabularies/
cp ../Sphana.Trainer/target/artifacts/llm/0.1.0/onnx/tokenizer.json Sphana.Database/Resources/Vocabularies/

# Option 2: Extract from tarball
tar -xzf ../Sphana.Trainer/target/manifests/latest.tar.gz -C Sphana.Database/Resources/Models/
```

### Step 5: Configure .NET Service

Update `appsettings.json` to point to the extracted models:

```json
{
  "Sphana": {
    "Models": {
      "EmbeddingModelPath": "Resources/Models/embedding.onnx",
      "RelationExtractionModelPath": "Resources/Models/relation_extraction.onnx",
      "GnnRankerModelPath": "Resources/Models/gnn_ranker.onnx",
      "LlmGeneratorModelPath": "Resources/Models/llm_generator.onnx",
      "UseGpu": true,
      "EmbeddingDimension": 384,
      "BatchSize": 32
    },
    "Tokenizer": {
      "VocabularyPath": "Resources/Vocabularies/vocab.json",
      "TokenizerConfigPath": "Resources/Vocabularies/tokenizer.json"
    }
  }
}
```

### Step 6: Run .NET Service

Start the Sphana Database service:

```bash
cd services/Sphana.Database
dotnet run --project Sphana.Database/Sphana.Database.csproj
```

The service will:
1. Load configuration from `appsettings.json`
2. Initialize ONNX Runtime with GPU provider
3. Load all ONNX models into memory
4. Create session pools for parallel inference
5. Start gRPC server on port 5001

## Manifest Structure

The manifest (`target/manifests/latest.json`) contains all model metadata:

```json
{
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "components": {
    "embedding": {
      "version": "0.1.0",
      "path": "embedding/0.1.0/onnx/embedding.onnx",
      "checksum": "sha256:abc123...",
      "metrics": {
        "val_cosine_sim": 0.872
      },
      "model_config": {
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "output_dimension": 384,
        "quantized": true
      }
    },
    "relation": {
      "version": "0.1.0",
      "path": "relation/0.1.0/onnx/relation_extraction.onnx",
      "checksum": "sha256:def456...",
      "metrics": {
        "val_macro_f1": 0.745
      },
      "model_config": {
        "base_model": "bert-base-uncased",
        "num_labels": 42,
        "quantized": true
      }
    },
    "gnn": {
      "version": "0.1.0",
      "path": "gnn/0.1.0/onnx/gnn_ranker.onnx",
      "checksum": "sha256:ghi789...",
      "metrics": {
        "val_ndcg_at_10": 0.812
      },
      "model_config": {
        "hidden_dim": 128,
        "num_layers": 3,
        "quantized": true
      }
    },
    "llm": {
      "version": "0.1.0",
      "path": "llm/0.1.0/onnx/llm_generator.onnx",
      "checksum": "sha256:jkl012...",
      "model_config": {
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "max_seq_length": 2048,
        "quantized": true
      }
    }
  },
  "system_requirements": {
    "onnx_runtime": ">=1.20.0",
    "cuda": ">=12.0",
    "cudnn": ">=9.0"
  }
}
```

## Automated Integration Workflow

Use the workflow command for end-to-end automation:

```bash
python -m sphana_trainer.cli workflow run \
    --ingest-config configs/ingest/wiki.yaml \
    --embedding-config configs/embedding/wiki.yaml \
    --relation-config configs/relation/wiki.yaml \
    --gnn-config configs/gnn/wiki.yaml \
    --llm-config configs/llm/base.yaml \
    --export-config configs/export/wiki.yaml \
    --package-config configs/export/wiki.yaml \
    --manifest target/manifests/latest.json \
    --build-datasets \
    --dataset-output-dir target/datasets/wiki
```

This command orchestrates:
1. Data ingestion
2. Dataset building
3. Model training (all components)
4. ONNX export with quantization
5. Manifest generation
6. Artifact packaging

## Model Hot-Reloading

The .NET service can hot-reload models without restart:

1. Place new ONNX files in `Resources/Models/`
2. Update manifest with new versions
3. Send reload signal via admin API:

```bash
curl -X POST http://localhost:5000/admin/reload-models
```

The service will:
- Load new models in background
- Run validation checks
- Swap to new models atomically
- Keep old models for in-flight requests

## Version Management

### List Available Versions

```bash
python -m sphana_trainer.cli artifacts list --artifact-root target/artifacts
```

Output:
```
Component: embedding
  v0.1.0 (2024-01-15) - val_cosine_sim: 0.872
  v0.2.0 (2024-01-20) - val_cosine_sim: 0.885 ← latest

Component: relation
  v0.1.0 (2024-01-15) - val_macro_f1: 0.745 ← latest
```

### Promote Specific Version

```bash
python -m sphana_trainer.cli artifacts promote embedding 0.2.0 \
    --artifact-root target/artifacts \
    --manifest target/manifests/production.json
```

### Rollback to Previous Version

```bash
python -m sphana_trainer.cli artifacts promote embedding 0.1.0 \
    --artifact-root target/artifacts \
    --manifest target/manifests/production.json
```

## Validation and Testing

### Parity Checks

Ensure ONNX models match PyTorch:

```bash
python -m sphana_trainer.cli artifacts parity-samples embedding 0.2.0
```

Output:
```
Running parity check for embedding v0.2.0...

Sample 1: Max diff = 0.0003 ✓
Sample 2: Max diff = 0.0002 ✓
Sample 3: Max diff = 0.0004 ✓

Average difference: 0.0003 (< 0.01 threshold)
Parity check PASSED ✓
```

### End-to-End Test

Test the full pipeline:

```bash
# Ingest test document
curl -X POST http://localhost:5001/ingest \
    -H "Content-Type: application/json" \
    -d '{
      "index": {"tenant_id": "test", "index_name": "docs"},
      "document": {"title": "Test", "document": "This is a test."}
    }'

# Query
curl -X POST http://localhost:5001/query \
    -H "Content-Type: application/json" \
    -d '{
      "index": {"tenant_id": "test", "index_name": "docs"},
      "query": "What is this about?"
    }'
```

## Troubleshooting

### Models Not Loading

**Error**: `Failed to load model from path: embedding.onnx`

**Solutions**:
1. Verify file exists at configured path
2. Check file permissions
3. Ensure ONNX Runtime version compatibility
4. Validate ONNX file integrity with `onnx.checker`

### CUDA Out of Memory

**Error**: `CUDA error: out of memory`

**Solutions**:
1. Reduce `BatchSize` in configuration
2. Use INT8 quantized models (not FP32)
3. Increase GPU memory
4. Fall back to CPU inference

### Parity Check Failures

**Error**: `Parity check failed: Max diff = 0.05`

**Solutions**:
1. Use higher precision during export (FP16 instead of INT8)
2. Increase calibration samples for quantization
3. Check for numerical instabilities in model
4. Adjust quantization parameters

## Best Practices

1. **Version Control**: Always version models with semantic versioning
2. **Testing**: Run parity checks before deploying new models
3. **Gradual Rollout**: Deploy to staging before production
4. **Monitoring**: Track model performance metrics in production
5. **Rollback Plan**: Keep previous model versions for quick rollback

## Next Steps

- Review [Model Components](/models) in detail
- Learn about [Deployment](/database-deploy) strategies
- Explore [Advanced Topics](/advanced) for optimization

