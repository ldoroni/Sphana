---
title: Model Components
description: Neural network models explained
---

# Model Components

Sphana uses four core neural network models to power its hybrid retrieval system. This page explains each component in detail.

## 1. Embedding Model

**Purpose**: Convert text into dense vector representations

**Architecture**: Transformer encoder (e.g., all-MiniLM-L6-v2)

**Input**: Text string (max 512 tokens)  
**Output**: 384-dimensional float vector

**Training**: Contrastive learning (SimCSE-style)
- Query-context pairs
- Temperature-scaled cosine similarity
- Batch negatives

**Usage**:
- Document chunk encoding
- Query encoding
- Semantic similarity search

**Performance**:
- Inference: ~5ms per batch (32 sequences)
- Throughput: ~6,000 seq/sec on GPU
- Model size: ~50MB (INT8)

## 2. Relation Extraction Model

**Purpose**: Extract entity-relationship triples from text

**Architecture**: BERT with classification head

**Input**: Text with entity markers  
Example: `[E1]John[/E1] works at [E2]Google[/E2]`

**Output**: Relation type (42 classes for TACRED)
- `per:employee_of`
- `org:founded_by`
- `per:city_of_birth`
- `no_relation`
- ...

**Training**: Supervised classification
- Entity-marked sequences
- Cross-entropy loss
- Macro-F1 optimization

**Usage**:
- Extract knowledge graph triples during ingestion
- Identify relationships between entities

**Performance**:
- Inference: ~10ms per batch (16 sequences)
- Throughput: ~1,600 seq/sec on GPU
- Model size: ~100MB (INT8)

## 3. GNN Ranker

**Purpose**: Rank candidate subgraphs by relevance

**Architecture**: Bi-directional Gated Graph Neural Network (GGNN)

**Input**: Subgraph with node features and edges
```
Nodes: [entity_embedding_1, entity_embedding_2, ...]
Edges: [(0, 1, forward), (1, 2, backward), ...]
```

**Output**: Relevance score (0.0 - 1.0)

**Training**: Listwise ranking (ListNet loss)
- Query + candidate subgraphs
- Pairwise ranking comparisons
- Gradient descent on ranking distribution

**Architecture Details**:
- **Layers**: 3 GNN layers
- **Hidden dim**: 128
- **Attention heads**: 4
- **Aggregation**: Max pooling
- **Readout**: Global attention

**Usage**:
- Re-rank subgraphs after hybrid retrieval
- Provide structure-aware relevance scores

**Performance**:
- Inference: ~12ms per batch (8 subgraphs)
- Throughput: ~650 subgraphs/sec on GPU
- Model size: ~80MB (INT8)

## 4. LLM Generator

**Purpose**: Generate natural language answers from context

**Architecture**: Decoder-only transformer (Llama-3.2-1B or Gemma-2-2B)

**Input**: Prompt with query + retrieved contexts
```
Query: What is backpropagation?

Context 1: Backpropagation is an algorithm...
Context 2: The chain rule is used...
Context 3: Gradients are computed...

Answer:
```

**Output**: Generated text (max 512 tokens)

**Training**: Instruction tuning (optional fine-tuning)
- Pre-trained on diverse text
- Optional: fine-tune on QA pairs
- Causal language modeling objective

**Model Options**:

| Model | Size | Memory (INT8) | Latency | Quality |
|-------|------|---------------|---------|---------|
| Qwen2-0.5B | 0.5B params | ~550MB | ~10ms | Good |
| Llama-3.2-1B | 1B params | ~1.2GB | ~15ms | Better |
| Gemma-2-2B | 2B params | ~2.4GB | ~25ms | Best |

**Usage**:
- Synthesize final answer from retrieved chunks
- Provide natural language responses

**Performance** (Llama-3.2-1B):
- Inference: ~15ms per generation
- Throughput: ~65 generations/sec on GPU
- Model size: ~1.2GB (INT8)

## Model Training Pipeline

### 1. Embedding Model Training

```python
# Dataset format
{
  "query": "What is a neural network?",
  "positive": "A neural network is a computational model...",
  "negative": "Python is a programming language..."  # Optional
}
```

**Loss Function**: Contrastive loss with in-batch negatives
```
L = -log(exp(sim(q, p) / τ) / Σ exp(sim(q, n) / τ))
```

**Metrics**:
- Validation cosine similarity
- Retrieval recall@K

### 2. Relation Extraction Training

```python
# Dataset format
{
  "text": "John works at Google in Mountain View.",
  "entity1": {"text": "John", "start": 0, "end": 4},
  "entity2": {"text": "Google", "start": 14, "end": 20},
  "label": "per:employee_of"
}
```

**Loss Function**: Cross-entropy loss
```
L = -Σ y_i log(ŷ_i)
```

**Metrics**:
- Macro-F1 score
- Precision/Recall per relation type

### 3. GNN Ranker Training

```python
# Dataset format
{
  "query_id": "q1",
  "candidates": [
    {
      "node_features": [[0.1, ...], [0.2, ...]],
      "edge_index": [[0, 1], [1, 2]],
      "edge_directions": [0, 1],
      "label": 1.0  # Ground truth relevance
    },
    ...
  ]
}
```

**Loss Function**: ListNet loss
```
L = -Σ y_i log(softmax(ŷ_i))
```

**Metrics**:
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)

### 4. LLM Generator Training

```python
# Dataset format (optional fine-tuning)
{
  "query": "What is backpropagation?",
  "context": "Backpropagation is...",
  "answer": "Backpropagation is the algorithm..."
}
```

**Loss Function**: Causal language modeling
```
L = -Σ log P(token_i | tokens_<i)
```

**Metrics**:
- Perplexity
- ROUGE score (for QA)

## ONNX Export

All models are exported to ONNX format for production:

```python
# PyTorch model
model = EmbeddingModel.from_pretrained(checkpoint)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "embedding.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["embeddings"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "embeddings": {0: "batch"}
    },
    opset_version=17
)

# Apply INT8 quantization
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "embedding.onnx",
    "embedding_int8.onnx",
    weight_type=QuantType.QInt8
)
```

**Benefits**:
- ~50% model size reduction
- ~2x inference speedup
- Cross-platform compatibility
- Hardware-optimized kernels

## Model Versioning

Models are versioned using semantic versioning:

```
embedding/
├── 0.1.0/
│   ├── pytorch/
│   │   └── checkpoint.pt
│   ├── onnx/
│   │   └── embedding.onnx
│   └── metadata.json
├── 0.2.0/
└── latest -> 0.2.0/
```

**Metadata** (`metadata.json`):
```json
{
  "version": "0.2.0",
  "component": "embedding",
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "metrics": {
    "val_cosine_sim": 0.872,
    "train_loss": 0.234
  },
  "training": {
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "dataset": "wiki-500k"
  },
  "export": {
    "onnx_version": "1.16.0",
    "quantized": true,
    "opset_version": 17
  }
}
```

## Model Selection Guidelines

### For Prototyping
- **Embedding**: all-MiniLM-L6-v2 (fast, good quality)
- **Relation**: bert-base-uncased (standard)
- **GNN**: 3 layers, 128 hidden dim
- **LLM**: Qwen2-0.5B (smallest, fastest)

### For Production
- **Embedding**: all-MiniLM-L6-v2 or mpnet-base-v2
- **Relation**: roberta-base (better accuracy)
- **GNN**: 4 layers, 256 hidden dim
- **LLM**: Gemma-2-2B (best quality/performance)

### For Research
- **Embedding**: Custom contrastive models
- **Relation**: Domain-specific fine-tuning
- **GNN**: Deeper networks, heterogeneous graphs
- **LLM**: Llama-3.2-3B or larger

## Next Steps

- Learn about [Training Workflow](/trainer-workflow)
- Explore [Integration Guide](/integration)
- Review [Configuration](/trainer-config) options

