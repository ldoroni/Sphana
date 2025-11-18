# Sphana Trainer Service

The Sphana Trainer is a Python 3.11 CLI that produces every neural artifact required by the Sphana NRDB: dense embedding encoders, entity-centric relation extractors, the GGNN reasoner, and the manifest/package consumed by the .NET gRPC runtime. Each command performs full PyTorch training (with Transformers + custom GNN), exports ONNX graphs, applies int8 quantization, and publishes metadata so downstream services can hot-load the latest versions.

## Highlights
- Typer-based CLI (`sphana-trainer`) with subcommands for `train embedding|relation|gnn`, `export`, `package`, and ingestion helpers.
- Ingestion orchestration via `ingest` plus `ingest-validate` to guarantee dataset quality before training, now with parser backends (`simple`, `spacy`, `stanza`) for relation extraction.
- spaCy/Stanza parsers and the optional Hugging Face relation classifier integrate with ingestion, and their parsed sentences are cached under each ingest cache for reproducible dataset builds.
- Trainer runtime supports mixed precision, checkpoint resume, and distributed data parallel (DDP) via `torchrun` (automatic device/scaler handling per component).
- Optional `metric_threshold` guards block artifact export if validation metrics regress (cosine similarity for embedding, macro-F1 for relation, validation loss for GNN).
- Structured metrics logging + dataset fingerprinting for every run, plus automatic pruning of old checkpoints.
- MLflow logging + the `train sweep` helper bring hyper-parameter searches under version control, while ONNX parity checks guarantee <1% drift relative to PyTorch outputs.
- Toggle `log_to_mlflow` in any training config (with optional `mlflow_tracking_uri`, `mlflow_experiment`, `mlflow_run_name`) to stream parameters/metrics to MLflow alongside on-disk logs.
- Artifact promotion CLI (`sphana-trainer artifacts ...`) to list, diff, and promote trained versions into manifests for the .NET runtime.
- YAML + Pydantic configuration (see `configs/`) with versioning, dataset overrides, and warmup/quantization knobs aligned with the design doc.
- Real training loops:
  - **Embedding:** contrastive SimCSE-style finetuning over JSONL query/context pairs, cosine evaluation, ONNX export + dynamic quantization.
  - **Relation Extraction:** entity-marked sequence classification with Hugging Face Transformers, macro-F1 early stopping, ONNX export.
  - **GNN Ranker:** listwise ListNet optimization over per-query candidate subgraphs, dynamic-axes ONNX export for the GGNN reasoner.
- Artifact registry: each run writes structured metadata + manifests under `target/artifacts/`, enabling the `.NET` service (and the `export`/`package` commands) to discover the correct ONNX payloads by version.
- Release pipeline: `sphana-trainer export` validates that all requested components are trained, composes the manifest defined in `src/sphana_trainer/schemas/manifests/model-manifest.schema.json`, and `sphana-trainer package` bundles the manifest plus ONNX binaries into a tarball ready for publication.

## Quick Start

### 1. Preconditions (run once per session)
```powershell
cd services/sphana-trainer
$env:PYTHONPATH="src"     # (PowerShell) ensures `python -m sphana_trainer.cli` can be imported

# Install dependencies
python -m venv .venv
.\.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip3 install -r .\requirements.txt
```

### 2. Train Models

#### Manual Per-Stage Commands
```powershell
# Train individual components (uses the bundled smoke datasets under src/tests/data)
python -m sphana_trainer.cli train embedding --config configs/embedding/base.yaml
python -m sphana_trainer.cli train relation  --config configs/relation/base.yaml
python -m sphana_trainer.cli train gnn       --config configs/gnn/base.yaml

# Produce manifest + tarball once training finishes
python -m sphana_trainer.cli export  --config configs/export/base.yaml
python -m sphana_trainer.cli package --config configs/export/base.yaml

# Preprocess raw documents + turn them into datasets
python -m sphana_trainer.cli dataset-download-wiki --titles-file samples/ai-ml-wiki-titles.small.txt --output target/data/wiki/docs.jsonl
python -m sphana_trainer.cli ingest --config configs/ingest/wiki.yaml
python -m sphana_trainer.cli dataset-build-from-ingest target/wiki/ingest --output-dir target/datasets/wiki

# Run ingestion validation (checks counts + files + relation schema)
python -m sphana_trainer.cli ingest-validate --config configs/ingest/base.yaml --stats `
    --chunks-schema src/sphana_trainer/schemas/ingestion/chunks.schema.json `
    --relations-schema src/sphana_trainer/schemas/ingestion/relations.schema.json

# Distributed training example (optional, requires >=2 GPUs + torchrun)
# torchrun --nproc_per_node=2 python -m sphana_trainer.cli train embedding --config configs/embedding/base.yaml --resume latest

# Manage artifacts
python -m sphana_trainer.cli artifacts list --artifact-root target/artifacts
python -m sphana_trainer.cli artifacts promote embedding 0.1.0 --artifact-root target/artifacts --manifest target/manifests/promoted.json
python -m sphana_trainer.cli artifacts bundle embedding 0.1.0 target/bundles/embedding --artifact-root target/artifacts

# Validate datasets before training
python -m sphana_trainer.cli dataset-validate src/tests/data/embedding/train.jsonl --type embedding
python -m sphana_trainer.cli dataset-stats src/tests/data/embedding/train.jsonl
```

#### Single Workflow Command  (training → export → ingestion → validation → artifact ops)
```powershell
python -m sphana_trainer.cli workflow run `
    --ingest-config configs/ingest/base.yaml `
    --embedding-config configs/embedding/base.yaml `
    --relation-config configs/relation/base.yaml `
    --gnn-config configs/gnn/base.yaml `
    --export-config configs/export/base.yaml `
    --package-config configs/export/base.yaml `
    --promote-component embedding `
    --promote-version 0.1.0 `
    --manifest target/manifests/latest.json `
    --build-datasets `
    --dataset-output-dir target/datasets/wiki
```

#### Wiki Workflow Commands
```powershell
# Run the high-fidelity Wiki workflow (spaCy/Stanza + MLflow logging)
python -m sphana_trainer.cli workflow wiki --artifact-root target/artifacts

# Launch an embedding sweep (results logged to MLflow automatically)
python -m sphana_trainer.cli train sweep embedding --config configs/embedding/base.yaml `
    --lr 2e-5 --lr 5e-5 `
    --batch-size 2 --batch-size 4 `
    --temperature 0.05 --temperature 0.07

# Review workflow progress
python -m sphana_trainer.cli workflow status --artifact-root target/artifacts

# Cache parser/classifier assets ahead of ingestion (optional: requires spaCy + Stanza extras)
# python -m sphana_trainer.cli ingest-cache-models --relation-model hf-internal-testing/tiny-random-bert --spacy-model en_core_web_sm --stanza-lang en
```

### Dataset Expectations
- **Ingestion (`ingest` command):** point `input_dir` at a directory of `.txt/.md/.json` files. The pipeline emits `chunks.jsonl` and `relations.jsonl` plus cached chunks under `artifact_root/cache`, and (for spaCy/Stanza) stores dependency parses under `cache/parses` for later inspection.
- Use `dataset-download-wiki` to download Wikipedia articles using titles from `samples/ai-ml-wiki-titles.*.txt`. The CLI downloads content into `target/data/wiki/docs.jsonl`, which is what `configs/ingest/wiki.yaml` consumes.
- Use `sphana-trainer dataset-build-from-ingest <ingest_dir>` to convert `chunks.jsonl`/`relations.jsonl` into training-ready splits for embedding, relation, and GNN models (outputs under `target/datasets` by default). Add curated corpora via `--extra-embedding`, `--extra-relation`, and `--extra-gnn` by providing paths to your sample JSONL files. Cached parse files are consumed automatically (override via `--parses-dir`).
- **Embedding (`train_file`, `validation_file`):** JSONL where each line contains `{"query": "...", "positive": "..."}` (additional keys like `anchor`/`context` are accepted). Validation is optional but recommended to monitor cosine similarity.
- **Relation (`train/validation`):** JSONL with `text`, `entity1`, `entity2`, and `label`. Each entity can specify `{"text": "...", "start": int, "end": int}`; spans are wrapped with `[E1]...[/E1]` and `[E2]...[/E2]` before tokenization.
- **GNN (`train/validation`):** JSONL where each record represents a query and contains `{"query_id": "...", "candidates": [{"node_features": [[...]], "edge_index": [[src, dst], ...], "edge_directions": [0/1], "label": float}, ...]}`. All candidates for a query are trained jointly via listwise loss.

Place datasets anywhere and override the paths via `dataset_path`, `train_file`, and `validation_file` in the YAML configs.

### Artifacts & Manifests
- Training commands write to `target/artifacts/<component>/<version>/` (version defaults to a timestamp if not provided) and update `target/artifacts/<component>/latest.json`.
- `export` reads the latest metadata for the requested components, generates `target/manifests/latest.json` (schema in `src/sphana_trainer/schemas/manifests/model-manifest.schema.json`), and records metrics + ONNX locations.
- `package` tars the manifest plus each component artifact into `<manifest>.tar.gz` so CI/CD can push a single file to blob storage.
- `artifacts parity-samples ...` emits input/output JSON for ONNX inference so the .NET gRPC service can run parity checks with its own runtime.

## Repository Layout
```
services/sphana-trainer
├── configs/                 # YAML configs (embedding/relation/gnn/export)
├── data/                    # Seed/public corpora checked into the repo
├── src/
│   ├── sphana_trainer/
│   │   ├── cli.py           # Typer entrypoint
│   │   ├── config.py        # Pydantic config models
│   │   ├── data/            # Dataset loaders
│   │   ├── exporters/       # ONNX export + quantization helpers
│   │   ├── models/          # Embedding encoder + GGNN implementations
│   │   ├── tasks/           # CLI task wrappers
│   │   ├── training/        # Training loops for each component
│   │   ├── schemas/         # JSON schemas (ingestion/datasets/manifests) consumed at runtime
│   │   └── utils/           # Metadata, seeding, filesystem helpers
│   └── tests/
│       ├── conftest.py
│       ├── data/            # Dummy datasets used by the tests
│       ├── e2e/             # Full pipeline tests
│       ├── integration/     # Ingestion integration tests
│       └── unit/            # All unit tests
├── target/                  # Generated outputs (artifacts, datasets, wiki downloads, logs)
├── requirements.txt
```

## Testing
- Fast unit tests (losses, utils): `pytest src/tests/unit/test_losses.py src/tests/unit/test_utils.py` (or `make unit`)
- Full pipeline (downloads the tiny HF model and runs ONNX export + quantization): `pytest src/tests/e2e/test_full_pipeline.py`
  - Expects CUDA 12.8 if you want to validate GPU execution, but also runs on CPU (slower).
- Run the whole suite (unit + e2e):
```bash
cd services/sphana-trainer
pytest  # or pytest -m "not slow" if you add markers later
```

- Ingestion-only tests (integration):
```bash
pytest -m ingestion  # focuses on ingestion + validation logic
```

## Notes
- Ensure CUDA 12.8 drivers are available before installing requirements (the file pins `--extra-index-url https://download.pytorch.org/whl/cu128` prior to `torch`).
- The CLI writes large checkpoints/ONNX models; set `SPHANA_WORKSPACE` to relocate log files if desired.
- For large datasets, prefer running inside a CUDA-enabled container/VM with >=1 A100 or similar to keep training times reasonable; configs can be adjusted for CPU-only smoke tests.
- Advanced ingestion parsers:
  - `parser: spacy` → `pip install spacy` + `python -m spacy download en_core_web_sm` (or your chosen model).
  - `parser: stanza` → `pip install stanza` + `python -m stanza.download en`.
- Distributed runs: launch with `torchrun --nproc_per_node=<gpus>` and set `ddp: true` plus `precision: fp16/bf16` in the component config to enable synchronized training. Non-primary ranks automatically skip artifact export and metadata writes.
- Quality gates: set `metric_threshold` within a component config to enforce minimum validation cosine/F1 or maximum GNN validation loss; runs without a validation split will raise if a threshold is requested.
- Use `src/sphana_trainer/schemas/ingestion/*.schema.json` with `ingest-validate --chunks-schema ... --relations-schema ...` to enforce data contracts on generated JSONL files.
- Dataset contracts are also available for training splits (`src/sphana_trainer/schemas/datasets/*.schema.json`); run `sphana-trainer dataset-validate` before training to catch malformed inputs.
- Structured metrics (`metrics.jsonl`) capture epoch timing, throughput, and device stats per run; dataset fingerprints and workflow state are stored under each artifact root so `workflow run` can resume or skip stages intelligently.
- For centralized experiment tracking, set `log_to_mlflow: true` (plus optional `mlflow_tracking_uri`, `mlflow_experiment`, `mlflow_run_name`) in any component config; the CLI handles initializing MLflow runs and logging hyperparameters/metrics from the primary rank. Use `train sweep` to iterate through small parameter grids without scripting.
- Enable profiling per component via `profile_steps: <N>` to capture PyTorch profiler traces stored alongside each run.
- After every `workflow run`, review `target/artifacts/workflow-report.json` for a consolidated summary (stage outputs, metrics, manifest) that can be attached to notebook runs or deployment PRs.
- `sphana-trainer workflow status` summarizes the latest successful stage per workspace, and `artifacts bundle` copies ONNX + metadata into a portable directory for manual delivery to the .NET team.
- Makefile shortcuts (`make unit`, `make integration`, `make wiki-train`, `make sweep-embedding`) simplify local workflows.
- Full offline documentation (HTML/CSS/JS) lives under [`docs/`](docs/index.html) for easy browsing in a browser.
