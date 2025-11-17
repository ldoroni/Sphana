## 1. Objectives and Scope
- Deliver two production-grade services that implement the Sphana NRDB design: (a) a Python CLI for training/quantizing/exporting all neural models to ONNX, and (b) a .NET 8 gRPC backend that ingests documents, maintains dual indexes (HNSW vector + PCSR KG), and answers free-text queries.
- Enforce the design mandates: 8-bit quantization, ONNX Runtime for every neural stage, hybrid vector+graph retrieval, listwise GNN ranking, and low-latency (<50 ms p95) targets.
- Provide a development workflow (repo layout, configuration, CI/CD, artifact management) that keeps model training artifacts, schema, and runtime deployments synchronized.

## 2. Cross-System Architecture Summary
- **Artifact flow:** Python CLI trains embedding/RE/GNN/LLM variants, quantizes, exports ONNX, and publishes versioned artifacts + metadata to an internal registry (e.g., Azure Blob + manifest JSON). The .NET service watches for approved versions via config service or feature flag to hot-reload models with zero downtime.
- **Data flow:** Ingestion → document chunking → embeddings → relation extraction → KG triple creation → persistence via PCSR + Parquet property store → vector index (HNSW/IVF) build/update → query pipeline combining ANN hits + KG traversal → GNN reranking → LLM summarization stream.
- **Shared contracts:** Protobuf definitions for query/index APIs; JSON/Parquet schemas for KG triples and embedding batches; YAML configs (hydra-style) describing model hyperparameters, dataset paths, and runtime overrides.
- **Observability:** Structured logging (OpenTelemetry) across both services, centralized metrics (Prometheus) for latency, GPU utilization, ANN recall, KG mutation throughput, and model drift indicators.

## 3. Python Training & ONNX Export CLI (Service A)
### 3.1 Responsibilities
- Manage datasets (TACRED, Re-TACRED, SemEval2010 Task8, custom corpora) and preprocessing (dependency parsing, entity-centric trees).
- Train/finetune:
  - Embedding encoder (e.g., all-MiniLM-L6-v2 or EmbeddingGemma) with domain adaptation.
  - Relation Extraction model (Bi-LSTM/Transformer + syntactic features).
  - GNN ranker (bi-directional GGNN with listwise loss).
  - Optional lightweight LLM head (Gemma 3.4B Q4_K_M) for answer generation conditioning.
- Quantize all weights/activations to int8, validate accuracy deltas, export to ONNX with optimum.exporters.
- Produce calibration metrics, regression tests, and configuration manifests for runtime service.

### 3.2 Architecture & Modules
- `sphana_train/cli.py`: entrypoint built with `typer` or `click`.
- `config/`: Hydra/OmegaConf config hierarchy (datasets, model hyperparams, optimizer, export options).
- `data_pipeline/`: ingestion, cleaning, dependency parser wrappers (Stanford Parser/Spacy), chunker (FastEmbed guidance).
- `models/`: modules per component (embedding_adapter.py, re_model.py, ggnn_ranker.py, llm_wrapper.py).
- `training/`: Trainer abstractions (Lightning/Fabric) with mixed precision, gradient accumulation, distributed support.
- `evaluation/`: metrics for F1 (RE), MRR/NDCG (retrieval), BLEU/ROUGE (LLM outputs), latency benchmarks.
- `export/`: ONNX export, quantization utilities (optimum, onnxruntime.quantization), validation harness to ensure ONNX parity vs. PyTorch weights.
- `artifacts/`: manifest writer (hashes, versions, dataset snapshot IDs) + uploader (Azure, S3, or local path).

### 3.3 CLI Commands (initial set)
- `train embedding --config configs/embedding/base.yaml`
- `train re-model --dataset tacred`
- `train gnn --triples data/triples.parquet --subgraphs data/ksg/`
- `evaluate <component> --checkpoint ...`
- `export onnx --component gnn --checkpoint ... --output artifacts/vX`
- `package release --manifest manifests/release.yaml` (bundles selected ONNX files + metadata).

### 3.4 Data & Pipeline Plan
- Implement deterministic preprocessing pipeline (hashing of raw docs, chunk IDs) to ensure reproducible KG nodes.
- Cache dependency parses and entity spans; store intermediate artifacts (Parquet) for reuse.
- Generate KG triples in training mode to bootstrap PCSR builder fixtures for the .NET team.
- Provide synthetic workloads for stress-testing (large graphs 10× RAM) by streaming edges via STXXL-compatible format.

### 3.5 Tooling & Dependencies
- Python 3.11, Poetry/uv for packaging.
- PyTorch 2.x + CUDA 12.x, Hugging Face Transformers/Sentence-Transformers, FastEmbed, Optimum ONNX, onnxruntime-gpu 1.20.x.
- SpaCy (or Stanza) for parsing; custom entity-centric tree builder.
- Weights & Biases or MLflow for experiment tracking (optional but recommended).

### 3.6 Testing & Quality Gates
- Unit tests for preprocessing, model components, and export scripts (pytest).
- Golden-batch inference parity: run ONNXRuntime vs. PyTorch on canonical inputs; fail if >1% divergence.
- Latency benchmarks per model to enforce <50 ms end-to-end target when combined.
- Contract tests ensuring manifest schema matches what the .NET service expects.

### 3.7 Deliverables
- Source tree under `services/sphana-trainer/`.
- Versioned ONNX artifacts + manifests.
- Comprehensive README with setup, hardware requirements, and troubleshooting (CUDA/ORT matrix).
- CI workflow (`github/workflows/train.yml`) running lint, unit tests, and dry-run exports on CPU.

## 4. .NET Core gRPC Indexing & Query Service (Service B)
### 4.1 Responsibilities
- Expose gRPC APIs for ingestion (`IndexDocument`, `BulkIngest`), maintenance (`RebuildIndexes`, `ReloadModels`), and querying (`Search`, `ExplainPath`, `HealthCheck`).
- Run ANN vector search + KG traversal using disk-resident PCSR layout with BFS-inspired block ordering.
- Execute ONNX Runtime inference for embeddings, RE, GNN ranker, and lightweight LLM head (streaming responses).
- Provide caching, batching, and monitoring to keep latency under target at production scale.

### 4.2 High-Level Architecture
- `.NET 8` WebHost with `Grpc.AspNetCore`.
- **Pipelines:**
  - **Ingestion Pipeline:** document normalizer → chunker (reuses FastEmbed for segmentation) → embedding inference (ONNXRuntime C#) → vector store (HNSW/IVF via Qdrant client or native HNSW.NET) → RE inference producing triples → KG builder writing PCSR structures + Parquet properties.
  - **Query Pipeline:** parse query (dependency parser via Python microservice or ONNX port) → question graph builder → hybrid retrieval (vector hits + entity lookup) → subgraph assembler → GNN ONNX inference for ranking → optional LLM ONNX inference for answer synthesis → gRPC streaming response.
- **Storage Layers:** 
  - Vector index (Qdrant, Milvus, or local HNSW library) with quantized 384-dim embeddings.
  - KG storage service implementing PCSR with contiguous disk pages + STXXL-style spillover.
  - Metadata DB (PostgreSQL) for documents, entity IDs, model versions, ingestion checkpoints.
  - Cache tier (Redis) for popular embeddings/subgraphs and GNN outputs.

### 4.3 gRPC Contract Plan (proto sketch)
- `service IndexService` with RPCs:
  - `IndexDocument(DocumentRequest) returns (IngestResponse)`
  - `BulkIngest(stream DocumentRequest) returns (IngestBatchResponse)`
  - `RebuildIndexes(RebuildRequest) returns (OperationStatus)`
- `service QueryService` with RPCs:
  - `Search(QueryRequest) returns (stream QueryResultChunk)` (supports streaming reasoning paths + final answer)
  - `ExplainPath(ExplainRequest) returns (ExplainResponse)`
  - `HealthCheck(google.protobuf.Empty) returns (HealthStatus)`
- Messages capture `model_version`, `tenant_id`, `ContextSnippet`, `GraphPath`, etc., aligning with manifests produced by Service A.

### 4.4 Model Runtime Integration
- Wrap ONNXRuntime C# sessions with warm pools; pre-load models per version.
- Implement dynamic reloading triggered by config or artifact watcher; ensure A/B rollouts via dual session pools.
- Expose batching layer: accumulate embedding inference requests (document chunks) within micro-batches (<5 ms wait) before dispatching to GPU.
- Provide CPU fallback path with degradation alerts if ORT fails to grab CUDA execution provider (per design risk 31–33).

### 4.5 Index & Storage Implementation Plan
- **Vector Index:** Baseline with in-process HNSW (hnswlib-dotnet) for MVP; optionally switch to Qdrant via gRPC for distributed deployments. Enforce embedding normalization + int8 storage.
- **KG Storage:** 
  - Build PCSR writer in .NET using memory-mapped files; maintain slack space for inserts.
  - Implement BFS-inspired block ordering offline job to reorder adjacency lists, minimizing random I/O.
  - Store entity/relation properties + embeddings in Parquet via `ParquetSharp`; expose columnar reads for batching.
- **External Memory Support:** Integrate with STXXL-like library or custom chunked vector to stream huge adjacency lists.

### 4.6 Performance, Scaling, and Ops
- Use Channels/Dataflow blocks for ingestion concurrency and backpressure.
- Track metrics: ingestion throughput, ANN recall vs. ground truth, KG traversal depth, GNN latency distribution.
- Implement request-level tracing (OpenTelemetry) to correlate query latency breakdown across ANN, KG, GNN, LLM.
- Horizontal scaling via Kubernetes: separate deployments for ingestion workers, query frontends, and background maintenance jobs (graph compaction, index rebuild).
- Nightly maintenance tasks: slack rebalance for PCSR, vector index compaction, cache refresh.

### 4.7 Testing Strategy
- Unit tests for protobuf mappers, KG builder, PCSR writer/reader, ONNX wrappers (xUnit).
- Integration tests spun up via docker-compose (Postgres, Redis, Qdrant) with seeded documents.
- Golden query suite referencing curated QA pairs; assert ranking quality (MRR, hit@k) and latency budgets.
- Chaos tests (fault injection) to verify CUDA fallback, cache eviction, and model reload safety.

### 4.8 Deliverables
- Source tree under `services/dotnet-grpc/`.
- Proto files + generated clients.
- Deployment manifests (Helm/Kustomize) and infrastructure-as-code (Bicep/Terraform) for cloud resources.
- Operations runbook documenting scaling knobs, cache warm-up, and troubleshooting.

## 5. Development Workflow, Repo Layout, and Tooling
- **Monorepo structure:**
  - `design/` (existing docs)
  - `services/sphana-trainer/` (CLI)
  - `services/dotnet-grpc/` (gRPC backend)
  - `protos/` (shared .proto + generated stubs)
  - `manifests/` (model + deployment configs)
  - `infra/` (IaC)
- **Branching & CI/CD:**
  - Feature branches → PR → GitHub Actions.
  - CI matrix: Python lint/tests, .NET build/tests, proto lint, manifest schema validation.
  - Nightly job to run training smoke tests on small dataset subset and regenerate synthetic KG fixtures.
- **Artifact/version management:**
  - Semantic versioning per model component (e.g., `embedding-1.2.0`).
  - Manifest linking model hashes, compatible CUDA/ORT versions, and config digest.
  - Promotion process from staging → production via signed manifests.
- **Dev Environments:**
  - Containerized dev (Dev Containers) with CUDA base image for Python; .NET container with ORT GPU packages.
  - Makefile or `justfile` shortcuts for setup, linting, running services locally.

## 6. Implementation Milestones
1. **Phase 0 – Foundations (Week 1):** finalize requirements, set up repo/CI, define proto + manifest schemas, provision shared infrastructure (artifact storage, databases).
2. **Phase 1 – Extraction Pipeline MVP (Weeks 2‑4):** implement Python preprocessing, train baseline embedding + RE models on public datasets, build CLI skeleton, export first ONNX artifacts, create sample KG + vector indexes.
3. **Phase 2 – gRPC Service MVP (Weeks 3‑6 overlap):** stand up .NET service with ingestion pipeline, vector store integration, basic KG persistence (initially CSR), simple query path without GNN.
4. **Phase 3 – GNN Reasoner & Listwise Ranking (Weeks 5‑8):** complete GNN training/export, integrate ONNX inference into .NET pipeline, add hybrid retrieval + subgraph assembly, implement streaming responses.
5. **Phase 4 – PCSR + Scale (Weeks 7‑10):** migrate storage to PCSR, implement BFS block reordering, add caching layers, run large-scale performance tests approaching 10× RAM graph sizes.
6. **Phase 5 – Hardening & Launch (Weeks 10‑12):** finalize observability, auto-scaling policies, disaster recovery drills, documentation, production-readiness review.

## 7. Risks & Mitigations
- **CUDA / ORT incompatibility:** lock versions per manifest, include pre-launch compatibility tests; keep CPU fallback with alerting.
- **PCSR complexity:** prototype with CSR, then incrementally add PCSR features; reuse academic references and build formal verification tests for insertion/deletion logic.
- **Model drift / dataset bias:** schedule recurring retraining via CLI, monitor retrieval quality metrics, integrate human-in-the-loop review for critical domains.
- **Latency creep:** enforce SLO budgets per pipeline stage; add autoscaling triggers on queue depth and GPU utilization.
- **Large graph I/O bottlenecks:** implement BFS-based layout early, use SSD NVMe, and benchmark STXXL streaming; consider sharding graphs by tenant/domain.

## 8. Acceptance Criteria
- Python CLI can train, evaluate, quantize, and export all required models with reproducible manifests and pass golden parity tests.
- .NET gRPC service ingests documents, maintains synchronized vector + KG indexes, handles hot model reloads, and answers benchmark queries within target latency and quality metrics.
- CI/CD pipeline enforces lint/tests, publishes artifacts, and deploys to staging environment automatically.
- Documentation covers setup, operations, schema contracts, and troubleshooting for both services.

