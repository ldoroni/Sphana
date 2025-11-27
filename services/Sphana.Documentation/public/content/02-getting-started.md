---
title: Getting Started
description: Install and run Sphana for the first time
---

# Getting Started

This guide will help you install and run Sphana Neural RAG Database for the first time.

## Prerequisites

### Required

- **Python 3.11+** - For training models
- **.NET 10.0 SDK** - For running the database service
- **CUDA 12.x + cuDNN 9.x** - For GPU acceleration (recommended)

### Optional

- **PostgreSQL 15+** - For metadata storage
- **Redis 7+** - For caching
- **Docker** - For containerized deployment

## Installation

### Step 1: Install Sphana Trainer (Python)

```bash
# Navigate to the trainer directory
cd services/Sphana.Trainer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train and Export Models

> **Note**: This step can take several hours depending on your hardware. For quick testing, use the small dataset samples included in the repository.

```bash
# Set PYTHONPATH (Windows PowerShell)
$env:PYTHONPATH="src"

# Train embedding model
python -m sphana_trainer.cli train embedding --config configs/embedding/base.yaml

# Train relation extraction model
python -m sphana_trainer.cli train relation --config configs/relation/base.yaml

# Train GNN ranker
python -m sphana_trainer.cli train gnn --config configs/gnn/base.yaml

# Optional: Train LLM generator
python -m sphana_trainer.cli train llm --config configs/llm/base.yaml

# Export all models to ONNX with INT8 quantization
python -m sphana_trainer.cli export --config configs/export/base.yaml

# Package models for deployment
python -m sphana_trainer.cli package --config configs/export/base.yaml
```

The trained models will be in `target/artifacts/` with the following structure:

```
target/artifacts/
├── embedding/
│   └── 0.1.0/
│       └── onnx/
│           └── embedding.onnx
├── relation/
│   └── 0.1.0/
│       └── onnx/
│           └── relation_extraction.onnx
├── gnn/
│   └── 0.1.0/
│       └── onnx/
│           └── gnn_ranker.onnx
└── llm/
    └── 0.1.0/
        └── onnx/
            └── llm_generator.onnx
```

### Step 3: Setup Sphana Database (.NET)

```bash
# Navigate to the database directory
cd services/Sphana.Database

# Restore NuGet packages
dotnet restore

# Build the project
dotnet build
```

### Step 4: Copy ONNX Models

Copy the trained ONNX models to the database's models directory:

```bash
# Create models directory
mkdir -p Sphana.Database/models

# Copy models (adjust paths as needed)
cp ../Sphana.Trainer/target/artifacts/embedding/0.1.0/onnx/embedding.onnx Sphana.Database/models/
cp ../Sphana.Trainer/target/artifacts/relation/0.1.0/onnx/relation_extraction.onnx Sphana.Database/models/
cp ../Sphana.Trainer/target/artifacts/gnn/0.1.0/onnx/gnn_ranker.onnx Sphana.Database/models/
# Optional:
cp ../Sphana.Trainer/target/artifacts/llm/0.1.0/onnx/llm_generator.onnx Sphana.Database/models/
```

### Step 5: Configure the Database

Update `appsettings.json` with your configuration:

```json
{
  "Sphana": {
    "Models": {
      "EmbeddingModelPath": "models/embedding.onnx",
      "RelationExtractionModelPath": "models/relation_extraction.onnx",
      "GnnRankerModelPath": "models/gnn_ranker.onnx",
      "LlmGeneratorModelPath": "models/llm_generator.onnx",
      "UseGpu": true,
      "EmbeddingDimension": 384,
      "BatchSize": 32
    },
    "VectorIndex": {
      "Dimension": 384,
      "HnswM": 16,
      "HnswEfConstruction": 200,
      "HnswEfSearch": 50,
      "StoragePath": "data/vector_index"
    },
    "KnowledgeGraph": {
      "GraphStoragePath": "data/knowledge_graph",
      "PcsrSlackRatio": 0.2,
      "MaxTraversalDepth": 3
    },
    "Query": {
      "TargetP95LatencyMs": 50,
      "VectorSearchWeight": 0.6,
      "GraphSearchWeight": 0.4,
      "VectorSearchTopK": 20
    }
  }
}
```

### Step 6: Run the Database Service

```bash
# Run the service
dotnet run --project Sphana.Database/Sphana.Database.csproj
```

The gRPC service will start on `https://localhost:5001` and HTTP on `http://localhost:5000`.

## Quick Test

### Verify Health

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "Healthy",
  "checks": {
    "vector_index": "Loaded",
    "graph_storage": "Loaded",
    "onnx_models": "Loaded"
  }
}
```

### Ingest a Document

Create a simple gRPC client or use `grpcurl`:

```bash
grpcurl -plaintext -d '{
  "index": {
    "tenant_id": "test",
    "index_name": "docs"
  },
  "document": {
    "title": "Neural Networks Basics",
    "document": "A neural network is a computational model inspired by biological neural networks. It consists of layers of interconnected nodes (neurons) that process information."
  }
}' localhost:5001 sphana.database.rpc.v1.SphanaDatabase/Ingest
```

### Query the Database

```bash
grpcurl -plaintext -d '{
  "index": {
    "tenant_id": "test",
    "index_name": "docs"
  },
  "query": "What is a neural network?"
}' localhost:5001 sphana.database.rpc.v1.SphanaDatabase/Query
```

## Docker Quick Start

Alternatively, use Docker Compose for a faster setup:

```bash
# Start all services
cd services/Sphana.Database
docker-compose up -d

# View logs
docker-compose logs -f sphana-database

# Stop services
docker-compose down
```

> **Note**: You still need to train models separately. The Docker setup expects ONNX models in the `models/` directory.

## Next Steps

Now that you have Sphana running:

1. **Learn the CLI**: Explore the [Trainer CLI Reference](/trainer-cli) for all available commands
2. **Understand the Architecture**: Read about [Database Architecture](/database-arch)
3. **API Integration**: Check out the [gRPC API documentation](/database-api)
4. **Training Pipeline**: Learn about the [Training Workflow](/trainer-workflow)

## Troubleshooting

### CUDA Not Available

If GPU is not detected:

1. Verify CUDA and cuDNN installation
2. Check ONNX Runtime GPU package compatibility
3. Set `"UseGpu": false` in configuration (falls back to CPU)

### Models Not Found

Ensure ONNX model files exist in the configured paths. Models must be:

1. Trained using the Trainer CLI
2. Exported to ONNX format with INT8 quantization
3. Compatible with ONNX Runtime 1.20.x

### Out of Memory

For large graphs or limited VRAM:

1. Use INT8 quantized models (not FP32)
2. Reduce `BatchSize` in configuration
3. Set `PreloadModelsInParallel: false`
4. Consider CPU inference for initial testing

### Port Already in Use

If port 5001 is already in use:

1. Change the port in `launchSettings.json`
2. Update firewall rules if needed
3. Check for other services using the same port

## Getting Help

- **Documentation**: Browse the complete [documentation](/overview)
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions and get support

