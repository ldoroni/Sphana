---
title: Advanced Topics
description: Performance optimization and troubleshooting
---

# Advanced Topics

This page covers advanced topics for optimizing performance, distributed training, and troubleshooting common issues.

## Performance Optimization

### INT8 Quantization

Reduce model size and improve inference speed:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="embedding.onnx",
    model_output="embedding_int8.onnx",
    weight_type=QuantType.QInt8,
    optimize_model=True
)
```

**Benefits**:
- ~50% model size reduction
- ~2x inference speedup on CPU
- ~1.5x speedup on GPU
- Minimal accuracy loss (<1%)

**Trade-offs**:
- Slightly lower precision
- Not all operators support INT8
- Requires calibration for static quantization

### Mixed Precision Training

Enable FP16 or BF16 training for faster convergence:

```yaml
training:
  precision: "bf16"  # or "fp16"
  gradient_checkpointing: true
  torch_compile: true  # PyTorch 2.0+
```

**FP16 vs BF16**:
- **FP16**: Faster, less memory, may have overflow issues
- **BF16**: Better for large models, requires Ampere GPU (A100, A10, etc.)

### Batch Processing

Optimize batch sizes for throughput:

```csharp
// .NET side: Batch incoming requests
var batchChannel = Channel.CreateBounded<InferenceRequest>(capacity: 100);

// Accumulate requests for ~5ms
var batch = new List<InferenceRequest>();
await foreach (var request in batchChannel.Reader.ReadAllAsync())
{
    batch.Add(request);
    if (batch.Count >= 32 || timeout)
    {
        var results = await RunBatchInference(batch);
        batch.Clear();
    }
}
```

**Guidelines**:
- Embedding: 32-64 per batch
- Relation: 16-32 per batch
- GNN: 4-8 per batch
- LLM: 1-4 per batch (memory-bound)

### GPU Optimization

Maximize GPU utilization:

```csharp
// Enable CUDA graph capture
var sessionOptions = new SessionOptions();
sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
sessionOptions.EnableCpuMemArena = false;
sessionOptions.EnableMemPattern = false;

// Use separate streams per session
sessionOptions.AppendExecutionProvider_CUDA(deviceId: 0, streamId: i);
```

**CUDA Best Practices**:
- Use TensorRT execution provider for NVIDIA GPUs
- Enable FP16 inference kernels
- Use pinned memory for host-device transfers
- Profile with `nvprof` or Nsight Systems

### Memory Optimization

Reduce memory footprint:

```csharp
// Pool inference sessions
var sessionPool = new ObjectPool<InferenceSession>(
    factory: () => CreateSession(modelPath),
    maxSize: 4
);

// Reuse tensors
var tensorPool = new TensorPool<float>(
    shape: new[] { 1, 512, 384 },
    poolSize: 32
);
```

**Memory Tips**:
- Use INT8 models instead of FP32
- Enable model sharing across sessions
- Limit concurrent inference requests
- Use memory-mapped I/O for large graphs

## Distributed Training

### Multi-GPU Training

Train on multiple GPUs with `torchrun`:

```bash
torchrun --nproc_per_node=4 \
    -m sphana_trainer.cli train embedding \
    --config configs/embedding/wiki.yaml \
    --resume latest
```

**Configuration**:
```yaml
training:
  ddp: true  # Enable DistributedDataParallel
  find_unused_parameters: false
  gradient_accumulation_steps: 4
```

**Scaling Efficiency**:
- 2 GPUs: ~1.9x speedup
- 4 GPUs: ~3.6x speedup
- 8 GPUs: ~6.8x speedup

### Multi-Node Training

For cluster training:

```bash
# Node 0 (master)
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    -m sphana_trainer.cli train embedding \
    --config configs/embedding/wiki.yaml

# Node 1
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    -m sphana_trainer.cli train embedding \
    --config configs/embedding/wiki.yaml
```

### DeepSpeed Integration

For very large models (LLM):

```python
from sphana_trainer.training.deepspeed_trainer import DeepSpeedLLMTrainer

trainer = DeepSpeedLLMTrainer(
    model=model,
    ds_config={
        "train_batch_size": 64,
        "gradient_accumulation_steps": 4,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"}
        }
    }
)
```

## Advanced Query Strategies

### Hybrid Weight Tuning

Adjust vector vs graph weights dynamically:

```csharp
public class AdaptiveHybridSearch
{
    private float CalculateVectorWeight(string query)
    {
        // Use more vector search for semantic queries
        if (IsSemanticQuery(query)) return 0.7f;
        
        // Use more graph search for entity-centric queries
        if (HasNamedEntities(query)) return 0.4f;
        
        return 0.6f;  // Default
    }
}
```

### Query Expansion

Expand queries with synonyms and related terms:

```python
def expand_query(query: str, embedding_model) -> List[str]:
    # Get query embedding
    query_emb = embedding_model.encode(query)
    
    # Find similar queries from history
    similar = index.search(query_emb, k=5)
    
    # Extract keywords
    expanded = [query]
    for result in similar:
        expanded.append(extract_keywords(result.text))
    
    return expanded
```

### Re-Ranking Strategies

Combine multiple signals:

```csharp
public float CalculateFinalScore(Candidate candidate)
{
    float vectorScore = candidate.VectorScore;
    float graphScore = candidate.GraphScore;
    float gnnScore = candidate.GnnScore;
    float recencyScore = CalculateRecency(candidate.Timestamp);
    float popularityScore = candidate.ViewCount / maxViews;
    
    return 0.4f * vectorScore
         + 0.3f * graphScore
         + 0.2f * gnnScore
         + 0.05f * recencyScore
         + 0.05f * popularityScore;
}
```

## Monitoring and Profiling

### PyTorch Profiler

Profile training code:

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for batch in dataloader:
        loss = train_step(batch)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("trace.json")
```

### ONNX Runtime Profiling

Profile inference:

```csharp
var sessionOptions = new SessionOptions();
sessionOptions.EnableProfiling = true;
sessionOptions.ProfileFileName = "onnx_profile.json";

var session = new InferenceSession(modelPath, sessionOptions);
// ... run inference ...
session.EndProfiling();
```

### OpenTelemetry Tracing

Add custom spans:

```csharp
using var activity = ActivitySource.StartActivity("HybridSearch");
activity?.SetTag("query", query);
activity?.SetTag("top_k", topK);

// Vector search span
using var vectorActivity = ActivitySource.StartActivity("VectorSearch");
var vectorResults = await vectorIndex.SearchAsync(queryEmbedding, topK);
vectorActivity?.SetTag("results_count", vectorResults.Count);

// Graph search span
using var graphActivity = ActivitySource.StartActivity("GraphTraversal");
var graphResults = await graphStorage.TraverseAsync(entities, depth: 3);
```

## Troubleshooting

### Training Issues

#### Loss Not Decreasing

**Symptoms**: Training loss stays flat or increases

**Solutions**:
1. Lower learning rate (divide by 10)
2. Increase warmup steps
3. Check data quality and labels
4. Verify model architecture
5. Add gradient clipping

#### Exploding Gradients

**Symptoms**: Loss becomes NaN, gradient norms very large

**Solutions**:
```yaml
training:
  max_grad_norm: 1.0  # Clip gradients
  learning_rate: 1e-5  # Lower LR
  precision: "fp32"  # Avoid FP16 overflow
```

#### Out of Memory

**Symptoms**: CUDA OOM during training

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Reduce max sequence length
5. Use mixed precision training

### Inference Issues

#### Slow Inference

**Symptoms**: High latency, low throughput

**Diagnostics**:
```csharp
var stopwatch = Stopwatch.StartNew();

var embeddingTime = MeasureTime(() => GetEmbedding(text));
var vectorSearchTime = MeasureTime(() => VectorSearch(embedding));
var graphTraversalTime = MeasureTime(() => GraphTraversal(entities));
var gnnRankingTime = MeasureTime(() => RankWithGNN(subgraphs));

Logger.LogInformation($"Embedding: {embeddingTime}ms");
Logger.LogInformation($"Vector: {vectorSearchTime}ms");
Logger.LogInformation($"Graph: {graphTraversalTime}ms");
Logger.LogInformation($"GNN: {gnnRankingTime}ms");
```

**Solutions**:
1. Use INT8 quantized models
2. Increase batch size
3. Enable GPU acceleration
4. Use TensorRT execution provider
5. Profile and optimize bottlenecks

#### Poor Result Quality

**Symptoms**: Irrelevant results, low accuracy

**Diagnostics**:
1. Check embedding quality (cosine similarities)
2. Verify relation extraction confidence scores
3. Inspect GNN ranking scores
4. Compare vector-only vs hybrid results

**Solutions**:
1. Retrain models with more data
2. Fine-tune on domain-specific data
3. Adjust hybrid search weights
4. Increase graph traversal depth
5. Add query expansion

### Data Quality Issues

#### Low Relation Extraction Accuracy

**Symptoms**: Few or incorrect relations extracted

**Solutions**:
1. Use better parser backend (spacy/stanza)
2. Lower `min_relation_confidence`
3. Fine-tune relation model on domain data
4. Manually label 100-1000 examples

#### Poor Embedding Quality

**Symptoms**: Low cosine similarity for relevant pairs

**Solutions**:
1. Fine-tune embedding model
2. Use domain-specific base model
3. Increase contrastive temperature
4. Add hard negatives to training data

## Security Considerations

### Model Security

- Validate ONNX files before loading
- Use checksums in manifest
- Scan for malicious operators
- Limit model file sizes

### API Security

- Implement rate limiting
- Add authentication (API keys, JWT)
- Use TLS for gRPC
- Validate all inputs

### Data Privacy

- Encrypt data at rest
- Use secure deletion
- Implement tenant isolation
- Add audit logging

## Next Steps

- Review [Getting Started](/getting-started) for basic setup
- Explore [Model Components](/models) in depth
- Check [API Reference](/database-api) for integration

