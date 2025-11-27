---
title: Database Architecture
description: Technical architecture and components
---

# Database Architecture

Sphana Database implements a hybrid architecture combining dense vector search with knowledge graph storage, all powered by ONNX Runtime for neural inference.

## System Overview

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ gRPC
       ▼
┌─────────────────────────────────────┐
│      Sphana Database Service        │
│  ┌─────────────────────────────┐   │
│  │   Document Ingestion        │   │
│  │   ┌─────────────────────┐   │   │
│  │   │ Chunking            │   │   │
│  │   │ Embedding           │◄──┼───┼── ONNX Runtime
│  │   │ Relation Extraction │   │   │      (GPU)
│  │   │ Graph Construction  │   │   │
│  │   └─────────────────────┘   │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │   Query Engine              │   │
│  │   ┌─────────────────────┐   │   │
│  │   │ Vector Search (HNSW)│   │   │
│  │   │ Graph Traversal     │   │   │
│  │   │ GNN Ranking         │◄──┼───┼── ONNX Runtime
│  │   │ LLM Generation      │   │   │      (GPU)
│  │   └─────────────────────┘   │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
       │         │         │
       ▼         ▼         ▼
┌──────────┐ ┌──────┐ ┌────────┐
│  Vector  │ │ PCSR │ │ Cache  │
│  Index   │ │ Graph│ │(Redis) │
└──────────┘ └──────┘ └────────┘
```

## Core Components

### 1. HNSW Vector Index

**Hierarchical Navigable Small World (HNSW)** provides fast approximate nearest neighbor search.

**Key Features**:
- **Multi-layer** graph structure
- **Logarithmic** search complexity O(log N)
- **Configurable** accuracy/speed trade-off
- **Persistent** storage with save/load

**Parameters**:
- `M` (16): Number of bi-directional links per node
- `efConstruction` (200): Quality during index building
- `efSearch` (50): Quality during search

**Memory Usage**:
- ~1.5KB per vector (384-dim, FP32)
- ~400 bytes per vector (384-dim, INT8)

**Performance**:
- Indexing: ~1,000 vectors/sec (CPU), ~10,000 vectors/sec (GPU)
- Search: <10ms p99 for 1M vectors

### 2. PCSR Graph Storage

**Packed Compressed Sparse Row (PCSR)** provides dynamic graph storage with efficient updates.

**Key Features**:
- **Dynamic** insertions and deletions (O(1) amortized)
- **Slack space** (20% by default) for fast updates
- **BFS traversal** up to configurable depth
- **Path finding** between entities

**Structure**:
```
Nodes: [e1, e2, e3, e4, ...]
Edges: [
  e1 → [e2, e3],      # e1's neighbors
  e2 → [e1, e4],      # e2's neighbors
  ...
]
Properties: Parquet/columnar storage
```

**Operations**:
- Add node: O(1)
- Add edge: O(1) amortized
- Traverse: O(V + E)
- Find path: O(b^d) where b=branching, d=depth

### 3. ONNX Runtime Infrastructure

All neural models run via **ONNX Runtime** with GPU acceleration.

**Model Session Pooling**:
```
SessionPool (size=4)
├─ Session 1 (GPU:0)
├─ Session 2 (GPU:0)
├─ Session 3 (GPU:0)
└─ Session 4 (GPU:0)
```

**Batching**:
- Requests are queued in channels
- Batches accumulate for ~5ms
- Executed together on GPU
- Results distributed to callers

**Features**:
- Automatic CPU fallback if GPU unavailable
- Dynamic batch sizing
- INT8 quantization support
- Tensor pooling for reduced allocation

### 4. Document Ingestion Service

**Pipeline**:

1. **Chunking** (512 tokens, 50 overlap)
   ```
   Document → Chunks (with overlap)
   ```

2. **Embedding Generation**
   ```
   Chunks → ONNX (Embedding) → Vectors (384-dim)
   ```

3. **Vector Index Update**
   ```
   Vectors → HNSW Index
   ```

4. **Named Entity Recognition**
   ```
   Chunks → NER → Entities
   ```

5. **Relation Extraction**
   ```
   (Entity1, Text, Entity2) → ONNX (RE) → Relations
   ```

6. **Graph Construction**
   ```
   (Entities, Relations) → PCSR Graph
   ```

**Configuration**:
```json
{
  "Ingestion": {
    "ChunkSize": 512,
    "ChunkOverlap": 50,
    "MaxConcurrency": 10,
    "MinRelationConfidence": 0.5
  }
}
```

### 5. Query Service

**Hybrid Retrieval Pipeline**:

1. **Query Embedding**
   ```
   Query → ONNX (Embedding) → Query Vector
   ```

2. **Vector Search**
   ```
   Query Vector → HNSW Index → Top-K Chunks
   ```

3. **Entity Extraction**
   ```
   Query → NER → Query Entities
   ```

4. **Subgraph Assembly**
   ```
   Query Entities + Vector Results → PCSR Graph → Subgraphs
   ```

5. **GNN Ranking**
   ```
   Subgraphs → ONNX (GNN) → Relevance Scores
   ```

6. **Result Fusion**
   ```
   (Vector Scores × 0.6) + (Graph Scores × 0.4) → Final Ranking
   ```

7. **LLM Generation** (optional)
   ```
   (Query + Top Results) → ONNX (LLM) → Answer
   ```

**Configuration**:
```json
{
  "Query": {
    "TargetP95LatencyMs": 50,
    "VectorSearchWeight": 0.6,
    "GraphSearchWeight": 0.4,
    "VectorSearchTopK": 20,
    "MaxSubgraphs": 10,
    "GnnEnabled": true
  }
}
```

## Storage Layer

### Vector Index Storage

**Format**: Custom binary format

```
Header (metadata)
├─ Dimension: 384
├─ Count: 1,234,567
├─ M: 16
└─ EfConstruction: 200

Layers (hierarchical)
├─ Layer 0 (all nodes)
├─ Layer 1 (subset)
└─ Layer 2 (subset)

Vectors (quantized)
├─ Node 0: [q1, q2, ..., q384]
├─ Node 1: [q1, q2, ..., q384]
└─ ...
```

### Knowledge Graph Storage

**Format**: PCSR + Parquet

```
graph.pcsr (adjacency structure)
├─ Node array
├─ Edge array
└─ Slack array

properties.parquet (node/edge properties)
├─ entity_id
├─ entity_text
├─ entity_type
├─ embedding (384-dim)
└─ confidence
```

## Performance Characteristics

### Latency Breakdown (Typical Query)

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Embedding | 5-10 | 20% |
| Vector Search | 2-5 | 10% |
| Graph Traversal | 5-10 | 20% |
| GNN Ranking | 10-15 | 30% |
| LLM Generation | 10-20 | 20% |
| **Total** | **~40ms** | **100%** |

### Throughput

- **Vector Search Only**: ~5,000 qps
- **Hybrid Search (no GNN)**: ~2,000 qps
- **Full Pipeline (with GNN)**: ~1,000 qps
- **With LLM**: ~500 qps

### Memory Usage

**With INT8 Quantization**:
- Embedding model: ~50MB
- Relation extraction: ~100MB
- GNN ranker: ~80MB
- LLM generator: ~1.2GB (Llama-3.2-1B) or ~550MB (Gemma-2-2B)
- Vector index: ~400 bytes × N vectors
- Graph storage: ~8 bytes × N nodes + ~16 bytes × M edges

**Total for 1M documents** (~5M chunks):
- Models: ~1.4GB
- Vector index: ~2GB
- Graph storage: ~500MB
- **Total**: ~4GB

## Scalability

### Vertical Scaling

- Increase GPU memory for larger batches
- Add more CPU cores for ingestion parallelism
- Use NVMe SSDs for faster graph I/O

### Horizontal Scaling

- **Shard by tenant**: Each tenant on separate instance
- **Shard by index**: Large indexes across multiple nodes
- **Load balancing**: Query frontends behind load balancer
- **Separation**: Dedicated ingestion workers + query frontends

### External Memory Support

For graphs exceeding RAM (10x+ RAM size):
- Memory-mapped files
- STXXL-style external memory
- BFS-ordered blocks for cache-friendly I/O

## Observability

### Metrics (Prometheus)

- `sphana_query_latency_ms` - Query latency histogram
- `sphana_ingestion_rate` - Documents/second
- `sphana_vector_index_size` - Number of indexed vectors
- `sphana_graph_node_count` - Knowledge graph nodes
- `sphana_onnx_inference_ms` - Per-model inference time
- `sphana_batch_size` - Current batch sizes

### Tracing (OpenTelemetry)

Each query produces a trace:
```
Query Span (50ms)
├─ Embedding (8ms)
├─ Vector Search (3ms)
├─ Graph Traversal (6ms)
├─ GNN Ranking (12ms)
└─ LLM Generation (18ms)
```

### Health Checks

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "Healthy",
  "checks": {
    "vector_index_loaded": true,
    "graph_storage_loaded": true,
    "embedding_model_loaded": true,
    "relation_model_loaded": true,
    "gnn_model_loaded": true,
    "llm_model_loaded": true,
    "gpu_available": true
  },
  "metadata": {
    "vector_count": 1234567,
    "graph_nodes": 456789,
    "graph_edges": 987654
  }
}
```

## Technology Stack

### Sphana Database (.NET)

- **.NET 10.0** - Runtime framework
- **ASP.NET Core** - Web framework
- **gRPC** - API protocol
- **ONNX Runtime 1.20.x** - Neural inference
- **BERTTokenizers** - Text tokenization
- **OpenTelemetry** - Observability

### Sphana Trainer (Python)

- **Python 3.11** - Language runtime
- **PyTorch 2.x** - ML framework
- **Transformers** - Pre-trained models
- **Typer** - CLI framework
- **MLflow** - Experiment tracking
- **ONNX** - Model export format

## Next Steps

- Learn about the [gRPC API](/database-api)
- Explore [Deployment](/database-deploy) options
- Understand [Model Components](/models)

