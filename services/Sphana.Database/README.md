# Sphana Database - Neural RAG Database (NRDB)

A high-performance, neural-augmented Retrieval-Augmented Generation (RAG) database system built on .NET 10.0, implementing hybrid vector and knowledge graph retrieval with ONNX Runtime inference.

## Overview

Sphana Database is an implementation of the Neural RAG Database (NRDB) architecture that combines:

- **Dense Vector Search** using HNSW (Hierarchical Navigable Small World) indexing
- **Knowledge Graph Storage** using PCSR (Packed Compressed Sparse Row) for dynamic graphs
- **Neural Inference** via ONNX Runtime with GPU acceleration
- **Hybrid Retrieval** combining semantic similarity and structured reasoning

## Architecture

The system implements the following neural components:

1. **Embedding Model** - Generates 384-dimensional dense vectors (e.g., all-MiniLM-L6-v2)
2. **Relation Extraction Model** - Extracts knowledge graph triples using entity-centric dependency trees
3. **GNN Ranker** - Bi-directional GGNN for subgraph ranking with listwise loss optimization
4. **LLM Generator** - Lightweight generation model (e.g., Gemma 3.4B quantized)

## Project Structure

```
services/Sphana.Database/
├── Sphana.Database/              # Main service project
│   ├── Configuration/            # Configuration models
│   ├── Infrastructure/
│   │   ├── Onnx/                # ONNX model wrappers
│   │   ├── VectorIndex/         # HNSW vector index
│   │   └── GraphStorage/        # PCSR graph storage
│   ├── Models/                   # Domain models
│   │   └── KnowledgeGraph/
│   ├── Services/                 # Application services
│   │   ├── Grpc/                # gRPC service implementations
│   │   ├── DocumentIngestionService.cs
│   │   └── QueryService.cs
│   └── Program.cs
├── Sphana.Database.Protos/       # Protobuf definitions
│   └── Protos/
│       └── sphana/
│           ├── common/v1/
│           └── database/v1/
└── Sphana.Database.Tests/        # Test project
    ├── Infrastructure/
    ├── Services/
    ├── Models/
    └── E2E/
```

## Prerequisites

### Required

- .NET 10.0 SDK or later
- ONNX Runtime 1.20.x
- CUDA 12.x + cuDNN 9.x (for GPU acceleration)

### Optional

- PostgreSQL 15+ (for metadata storage)
- Redis 7+ (for caching)
- Docker (for containerized deployment)

## Getting Started

### 1. Configuration

Copy and customize the configuration in `appsettings.json`:

```json
{
  "Sphana": {
    "Models": {
      "EmbeddingModelPath": "models/embedding.onnx",
      "RelationExtractionModelPath": "models/relation_extraction.onnx",
      "GnnRankerModelPath": "models/gnn_ranker.onnx",
      "UseGpu": true
    },
    "VectorIndex": {
      "Dimension": 384,
      "HnswM": 16,
      "StoragePath": "data/vector_index"
    },
    "KnowledgeGraph": {
      "GraphStoragePath": "data/knowledge_graph",
      "PcsrSlackRatio": 0.2
    }
  }
}
```

### 2. Model Preparation

ONNX models must be trained and exported using the Python training CLI (`sphana-trainer`). Place the exported models in the `models/` directory:

```
models/
├── embedding.onnx
├── relation_extraction.onnx
├── gnn_ranker.onnx
└── llm_generator.onnx (optional)
```

> **Note:** ONNX models are NOT included in this repository. They must be trained separately using the Python training pipeline in `services/Sphana.Trainer/`.

### 3. Build and Run

```bash
# Restore dependencies
dotnet restore

# Build the project
dotnet build

# Run the service
dotnet run --project Sphana.Database/Sphana.Database.csproj
```

The gRPC service will start on `https://localhost:5001`.

### 4. Run Tests

```bash
# Run all tests
dotnet test

# Run with coverage
dotnet test /p:CollectCoverage=true /p:CoverageReportsOutputFormat=lcov
```

> **Note:** Some tests require ONNX models to be available and are marked with `Skip` attribute. They will be enabled once models are provided.

## API Reference

### Ingest

Indexes a document into the database:

```protobuf
rpc Ingest (IngestRequest) returns (IngestResponse);

message IngestRequest {
  Index index = 1;
  Document document = 2;
}
```

### Query

Executes a hybrid query:

```protobuf
rpc Query (QueryRequest) returns (QueryResponse);

message QueryRequest {
  Index index = 1;
  string query = 2;
}
```

## Performance Targets

- **p95 Latency:** < 50ms for queries
- **Throughput:** > 1000 queries/second on GPU
- **Index Size:** Optimized for 10-100M documents
- **Memory:** < 4GB for core models (with quantization)

## Key Features

### Vector Search

- HNSW indexing with configurable M and efConstruction parameters
- Cosine, Euclidean, and Dot Product distance metrics
- Int8 quantization for 50% memory reduction
- Automatic embedding normalization

### Knowledge Graph

- Dynamic PCSR storage for efficient updates
- BFS-inspired layout for optimized disk I/O
- Multi-hop traversal up to configurable depth
- Path finding between entities

### Neural Inference

- ONNX Runtime with CUDA execution provider
- Batching with configurable wait times
- Session pooling for concurrent requests
- CPU fallback with warnings

### Observability

- OpenTelemetry metrics and tracing
- Prometheus metrics endpoint at `/metrics`
- Health check endpoint at `/health`
- Structured logging with configurable levels

## Development

### Adding New Features

1. Add domain models in `Models/`
2. Implement infrastructure in `Infrastructure/`
3. Create services in `Services/`
4. Add gRPC endpoints in `Services/Grpc/`
5. Write tests in `Sphana.Database.Tests/`

### Code Style

- Follow standard C# conventions
- Use nullable reference types
- Document public APIs with XML comments
- Keep classes sealed by default
- Prefer dependency injection

## Troubleshooting

### CUDA Not Available

If CUDA execution provider fails:

1. Verify CUDA 12.x and cuDNN 9.x are installed
2. Check ONNX Runtime GPU package version compatibility
3. Service will fall back to CPU with a warning

### Models Not Found

Ensure ONNX model files are in the configured paths. Models must be:

1. Trained using the Python training CLI
2. Exported to ONNX format with int8 quantization
3. Compatible with ONNX Runtime 1.20.x

### Memory Issues

For large graphs (>1M nodes):

1. Enable external memory support: `EnableExternalMemory: true`
2. Increase `MemoryLimitMb` in configuration
3. Consider distributed deployment

## Production Deployment

### Docker

```bash
docker build -t sphana-database .
docker run -p 5001:5001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  sphana-database
```

### Kubernetes

See `k8s/` directory for Helm charts and deployment manifests.

### Scaling

- **Horizontal:** Deploy multiple query frontends behind a load balancer
- **Vertical:** Increase GPU memory for larger batch sizes
- **Sharding:** Partition data by tenant or index

## Contributing

Please see the main repository CONTRIBUTING.md for guidelines.

## License

[Your License Here]

## References

Based on research in:

- Graph-augmented Learning to Rank for Querying
- QA-GNN: Reasoning with Language Models and Knowledge Graphs
- GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning
- Packed Compressed Sparse Row: A Dynamic Graph Representation

## Support

For issues and questions:
- GitHub Issues: [Link to issues]
- Documentation: [Link to docs]

