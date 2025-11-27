---
title: Overview
description: Introduction to Sphana Neural RAG Database
---

# Sphana: Neural RAG Database

## What is Sphana?

Sphana is a **Neural RAG Database (NRDB)** - a next-generation retrieval-augmented generation system that combines dense vector search with knowledge graph reasoning. Unlike traditional RAG systems that rely solely on vector similarity, Sphana explicitly models relationships between entities, enabling complex multi-hop reasoning and structured knowledge retrieval.

## Key Features

### 🔍 Hybrid Search
Combines **dense vector search** (HNSW) with **knowledge graph traversal** (PCSR) for superior retrieval quality. The system weighs semantic similarity (60%) and structured relationships (40%) by default.

### 🧠 Neural Components
- **Embedding Model**: 384-dim vectors (all-MiniLM-L6-v2)
- **Relation Extraction**: Entity-centric dependency tree extraction
- **GNN Ranker**: Bi-directional GGNN with listwise loss optimization
- **LLM Generator**: Lightweight models (Llama-3.2-1B or Gemma-2-2B)

### ⚡ High Performance
- **Sub-50ms** query latency (p95)
- **1000+ queries/second** on GPU
- **INT8 quantization** reduces model size by ~50%
- **Dynamic graphs** with efficient updates via PCSR

### 🎯 Production Ready
- **ONNX Runtime** with GPU acceleration (CUDA 12.x)
- **gRPC API** for high-performance communication
- **Docker & Kubernetes** deployment
- **OpenTelemetry** observability

## Architecture

Sphana consists of two main services:

### 1. Sphana Trainer (Python)
A comprehensive CLI tool for training and exporting neural models:

```bash
# Train models
python -m sphana_trainer.cli train embedding --config configs/embedding/base.yaml
python -m sphana_trainer.cli train relation --config configs/relation/base.yaml
python -m sphana_trainer.cli train gnn --config configs/gnn/base.yaml

# Export to ONNX with INT8 quantization
python -m sphana_trainer.cli export --config configs/export/base.yaml
```

### 2. Sphana Database (.NET)
A high-performance gRPC service for document ingestion and querying:

```csharp
// Ingest a document
var response = await client.IngestAsync(new IngestRequest {
    Index = new Index { TenantId = "tenant1", IndexName = "docs" },
    Document = new Document {
        Title = "Introduction to Neural Networks",
        Document_ = "Content here..."
    }
});

// Query the database
var result = await client.QueryAsync(new QueryRequest {
    Index = new Index { TenantId = "tenant1", IndexName = "docs" },
    Query = "What is a neural network?"
});
```

## How It Works

### Ingestion Pipeline

1. **Chunking**: Documents are split into semantic chunks
2. **Embedding**: Generate 384-dim vectors for each chunk
3. **Relation Extraction**: Extract entity-relationship triples
4. **Knowledge Graph**: Build graph structure with PCSR storage
5. **Indexing**: Create HNSW vector index for fast retrieval

### Query Pipeline

1. **Query Embedding**: Convert query to vector representation
2. **Hybrid Retrieval**: 
   - Vector search finds semantically similar chunks
   - Graph traversal finds structurally related entities
3. **GNN Ranking**: Rank candidate subgraphs using neural reasoner
4. **LLM Generation**: Synthesize final answer from top results

## Use Cases

### 📚 Technical Documentation
- Multi-document reasoning across API docs
- Code example retrieval with context
- Version-aware documentation search

### 🔬 Research Papers
- Cross-paper citation analysis
- Method and result extraction
- Related work discovery

### 🏢 Enterprise Knowledge Base
- Policy and procedure lookup
- Cross-department knowledge linking
- Compliance and audit trails

### 💡 Educational Content
- Concept relationship mapping
- Pre-requisite learning paths
- Multi-topic synthesis

## Why Sphana?

| Feature | Traditional RAG | Sphana NRDB |
|---------|----------------|-------------|
| **Search** | Vector only | Vector + Graph |
| **Reasoning** | Similarity-based | Structure + Similarity |
| **Multi-hop** | ❌ Limited | ✅ Native support |
| **Relationships** | ❌ Implicit | ✅ Explicit |
| **Latency** | ~10-20ms | <50ms (with GNN) |
| **Accuracy** | Good | Excellent |

## Performance Targets

- **Latency (p95)**: < 50ms
- **Throughput**: > 1,000 queries/second (GPU)
- **Index Size**: Optimized for 10-100M documents
- **Memory**: < 4GB for core models (with quantization)
- **GPU**: 2GB+ VRAM (INT8 models), 8GB+ (FP32 models)

## Quick Start

Ready to get started? Head over to the [Getting Started](/getting-started) guide to install and run Sphana for the first time.

> **Note**: Sphana requires trained ONNX models to operate. See the [Trainer CLI](/trainer-cli) documentation for model training instructions.

