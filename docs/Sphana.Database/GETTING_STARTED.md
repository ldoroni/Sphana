# Sphana Database - Quick Start Guide

This guide will help you get the Sphana Database up and running quickly.

## Prerequisites

### Required
- .NET 10.0 SDK ([Download](https://dotnet.microsoft.com/download/dotnet/10.0))
- Docker and Docker Compose (for containerized setup)

### Optional (for GPU acceleration)
- NVIDIA GPU with CUDA Compute Capability ≥ 7.0
- CUDA Toolkit 12.x
- cuDNN 9.x
- nvidia-docker runtime

## Quick Start (Docker)

The fastest way to get started is using Docker Compose:

### 1. Prepare ONNX Models

**⚠️ IMPORTANT:** ONNX models are required to run the application. They are NOT included in this repository.

You need to train and export models using the Python training CLI:

```bash
cd ../sphana-trainer
python -m sphana_train export onnx --component embedding --output ../Sphana.Database/models/
python -m sphana_train export onnx --component re-model --output ../Sphana.Database/models/
python -m sphana_train export onnx --component gnn --output ../Sphana.Database/models/
```

Your `models/` directory should contain:
```
models/
├── embedding.onnx
├── relation_extraction.onnx
├── gnn_ranker.onnx
└── llm_generator.onnx (optional)
```

### 2. Start the Services

```bash
# From the services/Sphana.Database directory
docker-compose up -d
```

This will start:
- Sphana Database (port 5001 for gRPC, 5000 for HTTP)
- PostgreSQL (port 5432)
- Redis (port 6379)
- Prometheus (port 9090)
- Grafana (port 3000)

### 3. Verify Health

```bash
# Check service health
curl http://localhost:5000/health

# View metrics
curl http://localhost:5000/metrics
```

### 4. Test with gRPC Client

Create a simple client to test the service:

```csharp
using Grpc.Net.Client;
using Sphana.Database.RPC.V1;

var channel = GrpcChannel.ForAddress("https://localhost:5001");
var client = new SphanaDatabase.SphanaDatabaseClient(channel);

// Ingest a document
var ingestRequest = new IngestRequest
{
    Index = new Index
    {
        TenantId = "tenant1",
        IndexName = "docs"
    },
    Document = new Document
    {
        Title = "Introduction to Neural Networks",
        Document_ = "Neural networks are computational models inspired by biological neural networks...",
        Metadata = { ["category"] = "AI" }
    }
};

var ingestResponse = await client.IngestAsync(ingestRequest);
Console.WriteLine($"Ingest status: {ingestResponse.Status.StatusCode}");

// Query
var queryRequest = new QueryRequest
{
    Index = new Index
    {
        TenantId = "tenant1",
        IndexName = "docs"
    },
    Query = "What are neural networks?"
};

var queryResponse = await client.QueryAsync(queryRequest);
Console.WriteLine($"Answer: {queryResponse.Result}");
```

## Quick Start (Local Development)

### 1. Start Dependencies

```bash
docker-compose up -d postgres redis
```

### 2. Configure Application

Update `appsettings.Development.json`:

```json
{
  "Sphana": {
    "Models": {
      "EmbeddingModelPath": "models/embedding.onnx",
      "UseGpu": false  // Set to true if you have CUDA
    },
    "PostgresConnectionString": "Host=localhost;Database=sphana;Username=postgres;Password=postgres",
    "RedisConnectionString": "localhost:6379"
  }
}
```

### 3. Run the Application

```bash
# Restore packages
dotnet restore

# Run migrations (if any)
# dotnet ef database update

# Run the service
dotnet run --project Sphana.Database/Sphana.Database.csproj
```

The service will start on:
- HTTP: http://localhost:5000
- HTTPS/gRPC: https://localhost:5001

### 4. Run Tests

```bash
# Run all tests
dotnet test

# Run specific test category
dotnet test --filter Category=Unit

# Run with coverage
dotnet test /p:CollectCoverage=true
```

**Note:** Some tests require ONNX models and are marked with `[Skip]`. They will run when models are available.

## Configuration

### Environment Variables

For Docker deployment:

```bash
# GPU support
export SPHANA_USE_GPU=true

# Database
export SPHANA_POSTGRES_CONNECTION="Host=postgres;Database=sphana;..."
export SPHANA_REDIS_CONNECTION="redis:6379"

# Model paths
export SPHANA_EMBEDDING_MODEL="/app/models/embedding.onnx"
```

### appsettings.json

Key configuration sections:

```json
{
  "Sphana": {
    "Models": {
      "UseGpu": true,
      "BatchSize": 32,
      "MaxBatchWaitMs": 5
    },
    "VectorIndex": {
      "HnswM": 16,
      "HnswEfSearch": 50
    },
    "Query": {
      "VectorSearchWeight": 0.6,
      "GraphSearchWeight": 0.4,
      "TargetP95LatencyMs": 50
    }
  }
}
```

## Troubleshooting

### ONNX Models Not Found

**Error:** `FileNotFoundException: ONNX model file not found`

**Solution:**
1. Ensure models are trained and exported from the Python CLI
2. Verify model paths in configuration
3. Check file permissions

### CUDA Not Available

**Error:** `Failed to enable CUDA execution provider`

**Solution:**
1. Verify CUDA and cuDNN are installed
2. Check ONNX Runtime GPU version compatibility
3. Service will fall back to CPU (check logs for warnings)

### Memory Issues

**Error:** `OutOfMemoryException` or slow performance

**Solution:**
1. Enable external memory: `"EnableExternalMemory": true`
2. Increase memory limits in Docker
3. Reduce batch sizes
4. Enable quantization

### Connection Refused

**Error:** `Connection refused` when accessing the service

**Solution:**
1. Check if the service is running: `docker-compose ps`
2. Verify port mappings: `docker-compose port sphana-database 5001`
3. Check firewall settings

## Monitoring

### Prometheus

Access Prometheus at http://localhost:9090

Key metrics:
- `sphana_query_latency_ms` - Query latency
- `sphana_ingestion_rate` - Documents ingested per second
- `sphana_vector_index_size` - Number of vectors indexed
- `sphana_graph_node_count` - Number of nodes in KG

### Grafana

Access Grafana at http://localhost:3000 (admin/admin)

Pre-configured dashboards (to be created):
- Query Performance
- Ingestion Pipeline
- System Resources
- Model Inference

### Logs

```bash
# View logs
docker-compose logs -f sphana-database

# Filter by level
docker-compose logs sphana-database | grep "ERROR"
```

## Next Steps

1. **Ingest Documents:** Start indexing your document collection
2. **Tune Parameters:** Adjust HNSW and query parameters for your use case
3. **Monitor Performance:** Use Prometheus/Grafana to track metrics
4. **Scale:** Deploy multiple instances behind a load balancer

## Additional Resources

- [README.md](README.md) - Full documentation
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Implementation details
- Design documents in `design/` directory
- Python training CLI docs at `../sphana-trainer/docs/index.html`

## Support

For issues and questions:
- Check existing issues on GitHub
- Review logs for error messages
- Consult IMPLEMENTATION.md for architecture details

---

**Remember:** The system requires trained ONNX models to operate. Without models, the service will start but ingestion and query operations will fail.

