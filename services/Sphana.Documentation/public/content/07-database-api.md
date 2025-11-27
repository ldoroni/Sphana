---
title: gRPC API Reference
description: Complete API reference and examples
---

# gRPC API Reference

Sphana Database exposes a high-performance gRPC API for document ingestion and querying.

## Service Definition

```protobuf
service SphanaDatabase {
  rpc Ingest (IngestRequest) returns (IngestResponse);
  rpc Query (QueryRequest) returns (QueryResponse);
  rpc Health (HealthCheckRequest) returns (HealthCheckResponse);
}
```

## Ingest API

### IngestRequest

Index a document into the database:

```protobuf
message IngestRequest {
  Index index = 1;
  Document document = 2;
}

message Index {
  string tenant_id = 1;
  string index_name = 2;
}

message Document {
  string document_id = 1;
  string title = 2;
  string document = 3;
  map<string, string> metadata = 4;
}
```

### IngestResponse

```protobuf
message IngestResponse {
  bool success = 1;
  string document_id = 2;
  int32 chunks_created = 3;
  int32 relations_extracted = 4;
  int64 processing_time_ms = 5;
}
```

### Example (gRPCurl)

```bash
grpcurl -plaintext -d '{
  "index": {
    "tenant_id": "acme",
    "index_name": "technical-docs"
  },
  "document": {
    "document_id": "doc-123",
    "title": "Introduction to Neural Networks",
    "document": "A neural network is a computational model...",
    "metadata": {
      "author": "John Doe",
      "date": "2024-01-15"
    }
  }
}' localhost:5001 sphana.database.rpc.v1.SphanaDatabase/Ingest
```

### Example (C#)

```csharp
using var channel = GrpcChannel.ForAddress("https://localhost:5001");
var client = new SphanaDatabase.SphanaDatabaseClient(channel);

var response = await client.IngestAsync(new IngestRequest
{
    Index = new Index
    {
        TenantId = "acme",
        IndexName = "technical-docs"
    },
    Document = new Document
    {
        DocumentId = "doc-123",
        Title = "Introduction to Neural Networks",
        Document_ = "A neural network is a computational model...",
        Metadata =
        {
            ["author"] = "John Doe",
            ["date"] = "2024-01-15"
        }
    }
});

Console.WriteLine($"Success: {response.Success}");
Console.WriteLine($"Chunks: {response.ChunksCreated}");
Console.WriteLine($"Relations: {response.RelationsExtracted}");
```

## Query API

### QueryRequest

```protobuf
message QueryRequest {
  Index index = 1;
  string query = 2;
  QueryOptions options = 3;
}

message QueryOptions {
  int32 top_k = 1;                // Default: 10
  float min_score = 2;            // Default: 0.0
  bool use_gnn = 3;               // Default: true
  bool generate_answer = 4;       // Default: false
  repeated string metadata_filters = 5;
}
```

### QueryResponse

```protobuf
message QueryResponse {
  repeated SearchResult results = 1;
  string generated_answer = 2;
  QueryMetadata metadata = 3;
}

message SearchResult {
  string document_id = 1;
  string chunk_id = 2;
  string text = 3;
  float score = 4;
  map<string, string> metadata = 5;
}

message QueryMetadata {
  int64 total_time_ms = 1;
  int64 vector_search_ms = 2;
  int64 graph_traversal_ms = 3;
  int64 gnn_ranking_ms = 4;
  int64 llm_generation_ms = 5;
  int32 candidates_retrieved = 6;
}
```

### Example (gRPCurl)

```bash
grpcurl -plaintext -d '{
  "index": {
    "tenant_id": "acme",
    "index_name": "technical-docs"
  },
  "query": "What is backpropagation?",
  "options": {
    "top_k": 5,
    "use_gnn": true,
    "generate_answer": false
  }
}' localhost:5001 sphana.database.rpc.v1.SphanaDatabase/Query
```

### Example (Python)

```python
import grpc
from sphana.database.rpc.v1 import sphana_database_pb2
from sphana.database.rpc.v1 import sphana_database_pb2_grpc

channel = grpc.insecure_channel('localhost:5001')
client = sphana_database_pb2_grpc.SphanaDatabaseStub(channel)

request = sphana_database_pb2.QueryRequest(
    index=sphana_database_pb2.Index(
        tenant_id="acme",
        index_name="technical-docs"
    ),
    query="What is backpropagation?",
    options=sphana_database_pb2.QueryOptions(
        top_k=5,
        use_gnn=True,
        generate_answer=False
    )
)

response = client.Query(request)

for result in response.results:
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.text[:200]}...")
    print("---")
```

## Health Check API

### HealthCheckRequest

```protobuf
message HealthCheckRequest {}
```

### HealthCheckResponse

```protobuf
message HealthCheckResponse {
  string status = 1;  // "Healthy", "Degraded", "Unhealthy"
  map<string, string> checks = 2;
  map<string, string> metadata = 3;
}
```

### Example

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "Healthy",
  "checks": {
    "vector_index_loaded": "true",
    "graph_storage_loaded": "true",
    "embedding_model_loaded": "true",
    "relation_model_loaded": "true",
    "gnn_model_loaded": "true",
    "gpu_available": "true"
  },
  "metadata": {
    "vector_count": "1234567",
    "graph_nodes": "456789",
    "graph_edges": "987654"
  }
}
```

## API Endpoint

Default endpoint: `https://localhost:5001`

Configure in `appsettings.json`:
```json
{
  "Kestrel": {
    "Endpoints": {
      "Grpc": {
        "Url": "https://localhost:5001",
        "Protocols": "Http2"
      },
      "Http": {
        "Url": "http://localhost:5000",
        "Protocols": "Http1"
      }
    }
  }
}
```

## Error Handling

gRPC status codes:

| Code | Description | Common Causes |
|------|-------------|---------------|
| OK | Success | - |
| INVALID_ARGUMENT | Invalid input | Missing required fields, invalid formats |
| NOT_FOUND | Resource not found | Index doesn't exist |
| ALREADY_EXISTS | Duplicate | Document ID already exists |
| RESOURCE_EXHAUSTED | Rate limit | Too many concurrent requests |
| INTERNAL | Server error | Model loading failure, CUDA errors |
| UNAVAILABLE | Service unavailable | Startup, maintenance |

Example error response:
```json
{
  "code": "INVALID_ARGUMENT",
  "message": "Document text cannot be empty",
  "details": []
}
```

## Rate Limiting

Default limits (configurable):
- **Ingest**: 100 requests/second per tenant
- **Query**: 1000 requests/second per tenant

Response headers:
- `X-RateLimit-Limit`: Maximum requests
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## Authentication

> **Note**: Currently not implemented. Planned for future releases.

Planned features:
- API key authentication
- JWT tokens
- mTLS for service-to-service

## Next Steps

- Explore [Deployment](/database-deploy) options
- Learn about [Integration](/integration) patterns
- Review [Database Architecture](/database-arch)

