"""Typer-based CLI entrypoint for the trainer service."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import itertools
import requests
import typer
import yaml
from rich.console import Console

from sphana_trainer import __version__
from sphana_trainer.artifacts import diff_artifacts, list_artifacts, promote_artifact, show_artifact
from sphana_trainer.artifacts.publisher import publish_manifest
from sphana_trainer.config import TrainerConfig, load_config
from sphana_trainer.data.dataset_builder import build_datasets_from_ingestion
from sphana_trainer.data.pipeline import (
    IngestionPipeline,
    load_ingest_config,
    summarize_ingestion,
    validate_ingestion_result,
    validate_with_schema,
)
from sphana_trainer.data.validation import dataset_statistics, validate_dataset_file
from sphana_trainer.logging import init_logging
from sphana_trainer.tasks import (
    EmbeddingTask,
    ExportTask,
    GNNTask,
    IngestionTask,
    PackageTask,
    RelationExtractionTask,
)
from sphana_trainer.workflow.state import (
    WorkflowLock,
    load_workflow_state,
    record_stage_failure,
    record_stage_start,
    record_stage_success,
    stage_is_current,
)
from sphana_trainer.workflow.report import generate_workflow_report


console = Console()
app = typer.Typer(help="Sphana neural training CLI", no_args_is_help=True, pretty_exceptions_enable=False)
train_app = typer.Typer(help="Train individual model components")
artifacts_app = typer.Typer(help="Inspect and promote artifacts")
workflow_app = typer.Typer(help="End-to-end workflow automation")
metrics_app = typer.Typer(help="Metrics and observability utilities")
profile_app = typer.Typer(help="Profiler trace helpers")
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
SCHEMA_ROOT = PACKAGE_ROOT / "schemas"
OUTPUT_ROOT = Path("target")
DEFAULT_ARTIFACT_ROOT = OUTPUT_ROOT / "artifacts"
DEFAULT_DATA_ROOT = OUTPUT_ROOT / "data"
DEFAULT_DATASETS_ROOT = OUTPUT_ROOT / "datasets"
DEFAULT_MLFLOW_PATH = (REPO_ROOT / "target" / "mlruns").resolve()
DEFAULT_MLFLOW_URI = DEFAULT_MLFLOW_PATH.as_uri()
DATASET_SCHEMAS = {
    "embedding": SCHEMA_ROOT / "datasets" / "embedding.schema.json",
    "relation": SCHEMA_ROOT / "datasets" / "relation.schema.json",
    "gnn": SCHEMA_ROOT / "datasets" / "gnn.schema.json",
}
# Parity sample files must be provided explicitly via command arguments
PARITY_SAMPLE_FILES: Dict[str, Optional[Path]] = {
    "embedding": None,
    "relation": None,
    "gnn": None,
}
app.add_typer(train_app, name="train")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(workflow_app, name="workflow")
app.add_typer(metrics_app, name="metrics")
app.add_typer(profile_app, name="profile")


def _normalize_tracking_uri(value: Optional[str], default_base: Optional[Path] = None) -> str:
    if value:
        if "://" in value:
            return value
        return Path(value).expanduser().resolve().as_uri()
    base = (default_base or DEFAULT_MLFLOW_PATH).expanduser().resolve()
    return base.as_uri()


@app.callback()
def main(
    ctx: typer.Context,
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Optional log file path."),
) -> None:
    """Initialize logging before executing any subcommand."""

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)

    target = log_file.expanduser().resolve() if log_file else None
    init_logging(target, log_level)


@app.command("version")
def version() -> None:
    """Print the CLI version."""

    console.print(f"sphana-trainer {__version__}")


@app.command("dataset-validate")
def dataset_validate(
    data: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, help="Path to dataset JSONL file."),
    dataset_type: Optional[str] = typer.Option(
        None,
        "--type",
        help="Dataset type (embedding|relation|gnn) to use built-in schema.",
        case_sensitive=False,
    ),
    schema: Optional[Path] = typer.Option(
        None,
        "--schema",
        help="Custom JSON schema path. Overrides --type when provided.",
        exists=True,
        file_okay=True,
    ),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Validate only the first N records."),
) -> None:
    """Validate a training dataset against the expected schema."""

    schema_path: Optional[Path] = None
    if schema:
        schema_path = schema
    elif dataset_type:
        key = dataset_type.lower()
        if key not in DATASET_SCHEMAS:
            raise typer.BadParameter(f"Unsupported dataset type '{dataset_type}'.")
        schema_path = DATASET_SCHEMAS[key]
    else:
        raise typer.BadParameter("Provide either --type or --schema.")

    try:
        validated = validate_dataset_file(data, schema_path, limit)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    console.print(
        f"[green]Validated {validated} records[/green] using schema [cyan]{schema_path}[/cyan]"
    )


@app.command("dataset-stats")
def dataset_stats_command(
    data: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, help="Path to dataset JSONL file."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Process only the first N records."),
) -> None:
    """Compute simple dataset statistics."""

    stats = dataset_statistics(data, limit)
    console.print_json(json.dumps(stats, indent=2))


@app.command("dataset-build-from-ingest")
def dataset_build_from_ingest(
    ingestion_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_dir: Path = typer.Option(
        DEFAULT_DATASETS_ROOT, "--output-dir", help="Directory to write derived datasets."
    ),
    min_confidence: float = typer.Option(
        0.2, "--min-confidence", min=0.0, max=1.0, help="Minimum relation confidence to keep."
    ),
    val_ratio: float = typer.Option(0.2, "--val-ratio", min=0.05, max=0.5, help="Validation split ratio."),
    seed: int = typer.Option(42, "--seed", help="Random seed for deterministic shuffling."),
    extra_embedding: List[Path] = typer.Option(
        [], "--extra-embedding", exists=True, file_okay=True, dir_okay=False, help="Additional embedding JSONL files."
    ),
    extra_relation: List[Path] = typer.Option(
        [], "--extra-relation", exists=True, file_okay=True, dir_okay=False, help="Additional relation JSONL files."
    ),
    extra_gnn: List[Path] = typer.Option(
        [], "--extra-gnn", exists=True, file_okay=True, dir_okay=False, help="Additional GNN JSONL files."
    ),
    parses_dir: Optional[Path] = typer.Option(
        None, "--parses-dir", exists=False, help="Optional directory containing cached parse JSON files."
    ),
) -> None:
    """Convert ingestion outputs (chunks/relations) into training datasets."""

    chunks_path = ingestion_dir / "chunks.jsonl"
    relations_path = ingestion_dir / "relations.jsonl"
    if not chunks_path.exists() or not relations_path.exists():
        raise typer.BadParameter("Expected chunks.jsonl and relations.jsonl in the ingestion directory.")
    if parses_dir is None:
        candidate = ingestion_dir / "cache" / "parses"
        if candidate.exists():
            parses_dir = candidate
    result = build_datasets_from_ingestion(
        chunks_path,
        relations_path,
        output_dir,
        val_ratio=val_ratio,
        min_confidence=min_confidence,
        seed=seed,
        extra_embedding=extra_embedding,
        extra_relation=extra_relation,
        extra_gnn=extra_gnn,
        parses_dir=parses_dir,
    )
    console.print(
        "[green]Derived datasets written to {dir}[/green]\n"
        "embedding train={et} val={ev}\n"
        "relation train={rt} val={rv}\n"
        "gnn train={gt} val={gv}".format(
            dir=result.output_dir,
            et=result.embedding_train,
            ev=result.embedding_val,
            rt=result.relation_train,
            rv=result.relation_val,
            gt=result.gnn_train,
            gv=result.gnn_val,
        )
    )


def _fetch_wiki_summary(session: requests.Session, title: str, *, attempts: int = 3, backoff: float = 1.5):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
    params = {"redirect": "true"}
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=params, timeout=10)
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(backoff * attempt)
            continue
        if response.status_code != 200:
            if attempt == attempts:
                return None
            time.sleep(backoff * attempt)
            continue
        data = response.json()
        text = data.get("extract")
        if not text:
            return None
        return {
            "id": data.get("pageid") or data.get("title") or title,
            "title": data.get("title") or title,
            "text": text,
            "source": "wikipedia",
        }
    return None


def _fetch_wiki_full_content(session: requests.Session, title: str, *, attempts: int = 3, backoff: float = 1.5):
    """Fetch full Wikipedia article content using the MediaWiki API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|pageprops",
        "explaintext": True,  # Plain text, no HTML
        "redirects": 1,
    }
    
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=params, timeout=30)
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(backoff * attempt)
            continue
        
        if response.status_code != 200:
            if attempt == attempts:
                return None
            time.sleep(backoff * attempt)
            continue
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        if not pages:
            return None
        
        # Get the first (and should be only) page
        page_id, page_data = next(iter(pages.items()))
        
        # Check if it's a missing page
        if page_id == "-1" or "missing" in page_data:
            return None
        
        text = page_data.get("extract", "")
        if not text:
            return None
        
        return {
            "id": int(page_id),
            "title": page_data.get("title", title),
            "text": text,
            "source": "wikipedia",
        }
    
    return None


@app.command("ingest-cache-models")
def ingest_cache_models(
    relation_model: Optional[str] = typer.Option(None, "--relation-model", help="Hugging Face relation model to cache."),
    spacy_model: Optional[str] = typer.Option(None, "--spacy-model", help="spaCy pipeline to download (e.g., en_core_web_sm)."),
    stanza_lang: Optional[str] = typer.Option(None, "--stanza-lang", help="Stanza language to download (e.g., en)."),
) -> None:
    """Download/cache models used by the ingestion pipeline."""

    if not any([relation_model, spacy_model, stanza_lang]):
        raise typer.BadParameter("Specify at least one of --relation-model, --spacy-model, or --stanza-lang.")

    if relation_model:
        _cache_relation_model(relation_model)
    if spacy_model:
        _download_spacy_model(spacy_model)
    if stanza_lang:
        _download_stanza(stanza_lang)


@app.command("dataset-download-wiki")
def dataset_download_wiki(
    output: Path = typer.Option(DEFAULT_DATA_ROOT / "wiki" / "docs.jsonl", "--output", help="Destination JSONL file."),
    title: List[str] = typer.Option([], "--title", help="Specific Wikipedia titles to download."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Maximum number of pages to fetch."),
    titles_file: Optional[Path] = typer.Option(
        None, "--titles-file", exists=True, file_okay=True, dir_okay=False, help="Optional file with titles (one per line)."
    ),
    shuffle: bool = typer.Option(True, "--shuffle/--no-shuffle", help="Shuffle the title list before downloading."),
    full_content: bool = typer.Option(False, "--full-content", help="Download full article content instead of summaries."),
) -> None:
    """Download Wikipedia articles into JSONL format. Requires --title or --titles-file."""

    titles = list(title)
    if titles_file:
        extra_titles = [line.strip() for line in titles_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        titles.extend(extra_titles)
    
    if not titles:
        raise typer.BadParameter(
            "No Wikipedia titles provided. Use --title to specify individual articles "
            "or --titles-file to provide a file with titles (one per line)."
        )
    
    if shuffle:
        import random

        random.shuffle(titles)
    if limit is not None:
        titles = titles[:limit]
    session = requests.Session()
    if hasattr(session, "headers"):
        session.headers.update({"User-Agent": "sphana-trainer/0.1 (+https://github.com/)"})
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fetched = 0
    
    mode = "full content" if full_content else "summaries"
    console.print(f"[cyan]Downloading {mode} for {len(titles)} titles...[/cyan]")
    
    with output.open("w", encoding="utf-8") as handle:
        for i, t in enumerate(titles, 1):
            try:
                if full_content:
                    record = _fetch_wiki_full_content(session, t)
                else:
                    record = _fetch_wiki_summary(session, t)
            except requests.RequestException as exc:
                console.print(f"[yellow]Skipping {t}: {exc}[/yellow]")
                continue
            if not record:
                console.print(f"[yellow]Skipping {t}: no content available[/yellow]")
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            fetched += 1
            if i % 10 == 0:
                console.print(f"[cyan]Progress: {fetched}/{i} successful ({len(titles) - i} remaining)[/cyan]")

    if fetched == 0:
        raise typer.BadParameter("No Wikipedia pages were downloaded.")
    console.print(f"[green]Saved {fetched} Wikipedia pages to {output}[/green]")


@train_app.command("embedding")
def train_embedding(
    config: Path = typer.Option(
        Path("configs/embedding/base.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        readable=True,
        help="Path to the embedding YAML config.",
    ),
    precision: Optional[str] = typer.Option(None, "--precision", help="Override precision, e.g. fp16."),
    gradient_accumulation: Optional[int] = typer.Option(
        None, "--grad-accum", min=1, help="Override gradient accumulation steps."
    ),
    profile_steps: Optional[int] = typer.Option(
        None, "--profile-steps", min=0, help="Capture profiler traces for N steps."
    ),
) -> None:
    """Train the embedding encoder."""

    embedding_cfg, artifact_root = _resolve_component_config(config, "embedding")
    if precision:
        embedding_cfg.precision = precision  # type: ignore[assignment]
    if gradient_accumulation:
        embedding_cfg.gradient_accumulation = gradient_accumulation
    if profile_steps is not None:
        embedding_cfg.profile_steps = profile_steps
    EmbeddingTask(embedding_cfg, artifact_root).run()


@train_app.command("relation")
def train_relation(
    config: Path = typer.Option(
        Path("configs/relation/base.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        readable=True,
        help="Path to the relation extraction YAML config.",
    ),
    precision: Optional[str] = typer.Option(None, "--precision", help="Override precision, e.g. bf16."),
    gradient_accumulation: Optional[int] = typer.Option(
        None, "--grad-accum", min=1, help="Override gradient accumulation steps."
    ),
    profile_steps: Optional[int] = typer.Option(
        None, "--profile-steps", min=0, help="Capture profiler traces for N steps."
    ),
) -> None:
    """Train the relation extraction model."""

    relation_cfg, artifact_root = _resolve_component_config(config, "relation")
    if precision:
        relation_cfg.precision = precision  # type: ignore[assignment]
    if gradient_accumulation:
        relation_cfg.gradient_accumulation = gradient_accumulation
    if profile_steps is not None:
        relation_cfg.profile_steps = profile_steps
    RelationExtractionTask(relation_cfg, artifact_root).run()


@train_app.command("gnn")
def train_gnn(
    config: Path = typer.Option(
        Path("configs/gnn/base.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        readable=True,
        help="Path to the GNN YAML config.",
    ),
    precision: Optional[str] = typer.Option(None, "--precision", help="Override precision."),
    gradient_accumulation: Optional[int] = typer.Option(
        None, "--grad-accum", min=1, help="Override gradient accumulation steps."
    ),
    profile_steps: Optional[int] = typer.Option(
        None, "--profile-steps", min=0, help="Capture profiler traces for N steps."
    ),
) -> None:
    """Train the GNN ranker."""

    gnn_cfg, artifact_root = _resolve_component_config(config, "gnn")
    if precision:
        gnn_cfg.precision = precision  # type: ignore[assignment]
    if gradient_accumulation:
        gnn_cfg.gradient_accumulation = gradient_accumulation
    if profile_steps is not None:
        gnn_cfg.profile_steps = profile_steps
    GNNTask(gnn_cfg, artifact_root).run()


@train_app.command("sweep")
def train_sweep(
    component: str = typer.Argument(..., help="Component to sweep (embedding|relation|gnn)"),
    config: Path = typer.Option(..., "--config", "-c", exists=True, file_okay=True, readable=True, help="Base YAML config."),
    learning_rate: List[float] = typer.Option([], "--lr", help="Learning rates to explore."),
    batch_size: List[int] = typer.Option([], "--batch-size", help="Batch sizes to explore."),
    temperature: List[float] = typer.Option([], "--temperature", help="Embedding temperature values."),
    hidden_dim: List[int] = typer.Option([], "--hidden-dim", help="GNN hidden dimensions."),
    grid_file: Optional[Path] = typer.Option(
        None, "--grid-file", exists=True, file_okay=True, dir_okay=False, help="YAML file describing sweep overrides."
    ),
    mlflow_tracking_uri: Optional[str] = typer.Option(
        None, "--mlflow-tracking-uri", help="Override MLflow tracking URI for sweep runs."
    ),
) -> None:
    """Run a small hyper-parameter sweep for a component."""

    component = component.lower()
    if component not in {"embedding", "relation", "gnn"}:
        raise typer.BadParameter("Component must be one of embedding|relation|gnn.")

    trainer_cfg = load_config(config)
    cfg = getattr(trainer_cfg, component)
    if cfg is None:
        raise typer.BadParameter(f"No '{component}' section found in {config}")
    artifact_root = trainer_cfg.artifact_root or trainer_cfg.workspace_dir
    grid_options = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    if component == "embedding":
        grid_options["temperature"] = temperature
    if component == "gnn":
        grid_options["hidden_dim"] = hidden_dim
    if grid_file:
        file_grid = _load_sweep_grid_file(grid_file)
        for key, values in file_grid.items():
            grid_options.setdefault(key, []).extend(values)
    overrides_list = _build_sweep_grid(grid_options)
    if not overrides_list:  # pragma: no cover - _build_sweep_grid always returns at least one combo
        overrides_list = [{}]
    console.print(f"[cyan]Launching {len(overrides_list)} sweep runs for {component}[/cyan]")
    sweep_base = Path(artifact_root) / "mlruns"
    candidate_uri = mlflow_tracking_uri or cfg.mlflow_tracking_uri or trainer_cfg.tracking_uri
    default_tracking_uri = _normalize_tracking_uri(candidate_uri, default_base=sweep_base)
    for idx, overrides in enumerate(overrides_list):
        run_cfg = cfg.model_copy(deep=True)
        for key, value in overrides.items():
            if value is None:  # pragma: no cover - grid values sanitized
                continue
            if not hasattr(run_cfg, key):  # pragma: no cover - grid only includes valid keys
                continue
            setattr(run_cfg, key, value)
        base_version = cfg.version or f"{component}-sweep"
        run_cfg.version = f"{base_version}-{idx:02d}"
        run_cfg.log_to_mlflow = True
        run_cfg.mlflow_tracking_uri = run_cfg.mlflow_tracking_uri or default_tracking_uri
        console.print(f"[green]Sweep {idx + 1}/{len(overrides_list)}[/green] overrides={overrides}")
        if component == "embedding":
            result = EmbeddingTask(run_cfg, artifact_root).run()
        elif component == "relation":
            result = RelationExtractionTask(run_cfg, artifact_root).run()
        else:
            result = GNNTask(run_cfg, artifact_root).run()
        _record_sweep_result(component, artifact_root, run_cfg, overrides, result)


@app.command("export")
def export_manifest(
    config: Path = typer.Option(
        Path("configs/export/base.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        readable=True,
        help="Path to the export YAML config.",
    )
) -> None:
    """Generate an artifact manifest."""

    export_cfg, artifact_root = _resolve_component_config(config, "export")
    ExportTask(export_cfg, artifact_root).run()


@app.command("package")
def package_release(
    config: Path = typer.Option(
        Path("configs/export/base.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        readable=True,
        help="Path to the export YAML config.",
    )
) -> None:
    """Bundle artifacts and manifest into a tarball."""

    export_cfg, artifact_root = _resolve_component_config(config, "export")
    PackageTask(export_cfg, artifact_root).run()


@app.command("ingest")
def ingest_documents(
    config: Path = typer.Option(
        Path("configs/ingest/base.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        readable=True,
        help="Path to the ingestion YAML config.",
    ),
    force: bool = typer.Option(False, "--force", help="Ignore cache and reprocess."),
) -> None:
    """Run the document ingestion pipeline."""

    ingest_cfg = load_ingest_config(config)
    task = IngestionTask(ingest_cfg)
    task.force = force
    task.run()


@app.command("ingest-validate")
def ingest_validate(
    config: Path = typer.Option(
        Path("configs/ingest/base.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        readable=True,
        help="Path to the ingestion YAML config.",
    ),
    force: bool = typer.Option(False, "--force", help="Re-run ingestion ignoring cache."),
    stats: bool = typer.Option(False, "--stats", help="Print ingestion statistics after validation."),
    chunks_schema: Optional[Path] = typer.Option(
        None, "--chunks-schema", help="Optional JSON schema to validate chunk records."
    ),
    relations_schema: Optional[Path] = typer.Option(
        None, "--relations-schema", help="Optional JSON schema to validate relation records."
    ),
) -> None:
    """Run ingestion and validate outputs."""

    ingest_cfg = load_ingest_config(config)
    pipeline = IngestionPipeline(ingest_cfg)
    result = pipeline.run(force=force)
    validate_ingestion_result(result)
    if chunks_schema or relations_schema:
        validate_with_schema(result, chunks_schema, relations_schema)
    console.print(
        f"[green]Ingestion valid:[/green] docs={result.document_count}, chunks={result.chunk_count}, relations={result.relation_count}"
    )
    if stats:
        summary = summarize_ingestion(result)
        console.print(f"[cyan]Labels:[/cyan] {summary['labels']}")


@artifacts_app.command("list")
def artifacts_list(
    component: Optional[str] = typer.Option(None, "--component", "-c", help="Component name to filter."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root directory."),
) -> None:
    entries = list_artifacts(artifact_root, component)
    if not entries:
        console.print("[yellow]No artifacts found.[/yellow]")
        return
    for comp, comps in entries.items():
        if not comps:
            console.print(f"[yellow]{comp}: no artifacts found[/yellow]")
            continue
        for meta in comps:
            console.print(f"{comp} -> {meta.version} (onnx={meta.onnx_path})")


@artifacts_app.command("show")
def artifacts_show(
    component: str = typer.Argument(..., help="Component name."),
    version: str = typer.Argument(..., help="Version to inspect."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root directory."),
) -> None:
    meta = show_artifact(component, version, artifact_root)
    console.print_json(json.dumps(meta.__dict__, indent=2))


@artifacts_app.command("diff")
def artifacts_diff_cli(
    component: str = typer.Argument(..., help="Component name."),
    version_a: str = typer.Argument(..., help="Reference version."),
    version_b: str = typer.Argument(..., help="Compare version."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root directory."),
) -> None:
    meta_a = show_artifact(component, version_a, artifact_root)
    meta_b = show_artifact(component, version_b, artifact_root)
    diff = diff_artifacts(meta_a, meta_b)
    console.print_json(json.dumps(diff, indent=2))


@artifacts_app.command("promote")
def artifacts_promote(
    component: str = typer.Argument(..., help="Component name."),
    version: str = typer.Argument(..., help="Version to promote."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root directory."),
    manifest: Optional[Path] = typer.Option(
        None, "--manifest", help="Optional manifest file to update after promotion."
    ),
    publish_url: Optional[str] = typer.Option(
        None, "--publish-url", help="Optional URL of the .NET service to publish manifests to."
    ),
) -> None:
    metadata_path = promote_artifact(component, version, artifact_root)
    console.print(f"[green]Promoted {component} {version} -> {metadata_path}[/green]")
    if manifest:
        meta = show_artifact(component, version, artifact_root)
        _update_manifest(manifest, component, meta)
        console.print(f"[blue]Manifest updated at {manifest}[/blue]")
        if publish_url:
            publish_manifest(manifest, publish_url)
            console.print(f"[cyan]Manifest published to {publish_url}[/cyan]")
    elif publish_url:
        raise typer.BadParameter("Publishing requires --manifest to be set.")


@artifacts_app.command("bundle")
def artifacts_bundle(
    component: str = typer.Argument(..., help="Component name."),
    version: str = typer.Argument(..., help="Version to bundle."),
    target: Path = typer.Argument(..., help="Target directory for bundled artifacts."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root directory."),
) -> None:
    """Copy ONNX/metadata files into a portable bundle directory."""

    meta = show_artifact(component, version, artifact_root)
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    def _copy(file_path: str) -> None:
        if not file_path:
            return
        src = Path(file_path)
        if src.exists():
            shutil.copy2(src, target / src.name)

    _copy(meta.onnx_path)
    if meta.quantized_path and meta.quantized_path != meta.onnx_path:
        _copy(meta.quantized_path)
    metadata_file = Path(meta.checkpoint_dir) / "metadata.json"
    if metadata_file.exists():
        shutil.copy2(metadata_file, target / f"{component}-{version}-metadata.json")
    console.print(f"[green]Bundled {component} {version} into {target}[/green]")


@artifacts_app.command("parity-samples")
def artifacts_parity_samples(
    component: str = typer.Argument(..., help="Component (embedding|relation|gnn)."),
    sample_file: Path = typer.Argument(..., exists=True, readable=True, help="JSONL file containing sample inputs."),
    output: Path = typer.Argument(..., help="Path to write the parity report JSON."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root directory."),
) -> None:
    """Generate ONNX parity samples consumable by the .NET service."""

    from sphana_trainer.artifacts.parity import generate_parity_sample

    report = generate_parity_sample(component.lower(), sample_file, artifact_root)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    console.print(f"[green]Parity sample saved to {output}[/green]")


def _should_run_stage(
    state: dict,
    stage: str,
    enabled: bool,
    output_path: Optional[Path],
    force: bool,
    force_set: set[str],
) -> bool:
    if not enabled:
        return False
    if force or stage in force_set:
        return True
    return not stage_is_current(state, stage, output_path)


@workflow_app.command("run")
def workflow_run(
    ingest_config: Optional[Path] = typer.Option(
        None, "--ingest-config", help="Path to ingestion config.", exists=True, file_okay=True
    ),
    build_datasets: bool = typer.Option(False, "--build-datasets", help="Derive training datasets from ingestion output."),
    dataset_output_dir: Optional[Path] = typer.Option(
        None, "--dataset-output-dir", help="Where to write derived datasets."
    ),
    dataset_min_confidence: float = typer.Option(
        0.2, "--dataset-min-confidence", min=0.0, max=1.0, help="Minimum relation confidence for dataset builder."
    ),
    dataset_val_ratio: float = typer.Option(
        0.2, "--dataset-val-ratio", min=0.05, max=0.5, help="Validation split ratio for dataset builder."
    ),
    dataset_seed: int = typer.Option(42, "--dataset-seed", help="Random seed for dataset builder shuffling."),
    embedding_config: Optional[Path] = typer.Option(
        None, "--embedding-config", help="Embedding trainer config.", exists=True, file_okay=True
    ),
    relation_config: Optional[Path] = typer.Option(
        None, "--relation-config", help="Relation trainer config.", exists=True, file_okay=True
    ),
    gnn_config: Optional[Path] = typer.Option(
        None, "--gnn-config", help="GNN trainer config.", exists=True, file_okay=True
    ),
    export_config: Optional[Path] = typer.Option(
        None, "--export-config", help="Export config.", exists=True, file_okay=True
    ),
    package_config: Optional[Path] = typer.Option(
        None, "--package-config", help="Package config.", exists=True, file_okay=True
    ),
    promote_component: Optional[str] = typer.Option(None, "--promote-component", help="Component to promote."),
    promote_version: Optional[str] = typer.Option(None, "--promote-version", help="Version to promote."),
    manifest: Optional[Path] = typer.Option(
        None, "--manifest", help="Manifest path for promotion.", file_okay=True
    ),
    publish_url: Optional[str] = typer.Option(None, "--publish-url", help="Optional service URL to publish manifest."),
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root for promotion."),
    mlflow_tracking_uri: Optional[str] = typer.Option(
        None,
        "--mlflow-tracking-uri",
        help="MLflow tracking URI (defaults to target/mlruns).",
    ),
    promote_publish: bool = typer.Option(False, "--publish", help="Publish manifest after promotion."),
    force: bool = typer.Option(False, "--force", help="Re-run all stages regardless of state"),
    force_stage: List[str] = typer.Option([], "--force-stage", help="Stage names to force re-run"),
    force_lock: bool = typer.Option(False, "--force-lock", help="Override the active workflow lock if present."),
) -> None:
    tracking_uri = _normalize_tracking_uri(mlflow_tracking_uri)
    report_path = run_workflow(
        ingest_config=ingest_config,
        build_datasets=build_datasets,
        dataset_output_dir=dataset_output_dir,
        dataset_min_confidence=dataset_min_confidence,
        dataset_val_ratio=dataset_val_ratio,
        dataset_seed=dataset_seed,
        embedding_config=embedding_config,
        relation_config=relation_config,
        gnn_config=gnn_config,
        export_config=export_config,
        package_config=package_config,
        promote_component=promote_component,
        promote_version=promote_version,
        manifest=manifest,
        publish_url=publish_url,
        artifact_root=artifact_root,
        promote_publish=promote_publish,
        force=force,
        force_stage=force_stage,
        force_lock=force_lock,
        mlflow_tracking_uri=tracking_uri,
    )
    console.print(f"[cyan]Workflow report saved to {report_path}[/cyan]")


@workflow_app.command("wiki")
def workflow_wiki(
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root directory."),
    parity: bool = typer.Option(True, "--parity/--no-parity", help="Generate parity fixtures after completion."),
    force_lock: bool = typer.Option(False, "--force-lock", help="Override the active workflow lock if present."),
    mlflow_tracking_uri: Optional[str] = typer.Option(
        None,
        "--mlflow-tracking-uri",
        help="MLflow tracking URI (defaults to target/mlruns).",
    ),
) -> None:
    ingest_cfg = REPO_ROOT / "configs" / "ingest" / "wiki.yaml"
    dataset_dir = DEFAULT_DATASETS_ROOT / "wiki"
    embedding_cfg = REPO_ROOT / "configs" / "embedding" / "wiki.yaml"
    relation_cfg = REPO_ROOT / "configs" / "relation" / "wiki.yaml"
    gnn_cfg = REPO_ROOT / "configs" / "gnn" / "wiki.yaml"
    manifest_path = artifact_root / "manifests" / "wiki-latest.json"
    tracking_uri = _normalize_tracking_uri(mlflow_tracking_uri)
    report_path = run_workflow(
        ingest_config=ingest_cfg,
        build_datasets=True,
        dataset_output_dir=dataset_dir,
        dataset_min_confidence=0.3,
        dataset_val_ratio=0.2,
        dataset_seed=42,
        embedding_config=embedding_cfg,
        relation_config=relation_cfg,
        gnn_config=gnn_cfg,
        export_config=None,
        package_config=None,
        promote_component=None,
        promote_version=None,
        manifest=manifest_path,
        publish_url=None,
        artifact_root=artifact_root,
        promote_publish=False,
        force=False,
        force_stage=[],
        force_lock=force_lock,
        mlflow_tracking_uri=tracking_uri,
    )
    console.print(f"[cyan]Wiki workflow report saved to {report_path}[/cyan]")
    if parity:
        parity_dir = artifact_root / "parity"
        generated = _generate_parity_bundle(artifact_root, parity_dir)
        if generated:
            console.print(f"[green]Parity fixtures written to {parity_dir}[/green]")


@workflow_app.command("status")
def workflow_status(
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root directory."),
) -> None:
    """Show workflow stage status."""

    state_path = artifact_root.expanduser().resolve() / "workflow-state.json"
    state = load_workflow_state(state_path)
    stages = state.get("stages", {})
    if not stages:
        console.print("[yellow]No workflow state recorded yet.[/yellow]")
        return
    console.print(f"[cyan]Workflow state ({state_path}):[/cyan]")
    for stage, info in stages.items():
        output = info.get("output") or "-"
        status = info.get("status", "unknown")
        finished = info.get("finished_at") or info.get("timestamp") or "-"
        line = f" - {stage}: {status} (output={output}, finished={finished})"
        if info.get("error"):
            line += f" [red]error={info['error']}[/red]"
        console.print(line)


@metrics_app.command("summarize")
def metrics_summarize(
    metrics_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to metrics.jsonl or a run directory containing metrics.jsonl.",
    )
) -> None:
    """Summarize metrics JSONL files."""

    metrics_path = metrics_path.expanduser().resolve()
    if metrics_path.is_dir():
        metrics_path = metrics_path / "metrics.jsonl"
    if not metrics_path.exists():
        raise typer.BadParameter(f"Metrics file not found at {metrics_path}")
    summary = _summarize_metrics(metrics_path)
    console.print_json(json.dumps(summary, indent=2))


@profile_app.command("traces")
def profile_traces(
    artifact_root: Path = typer.Option(DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Artifact root to scan."),
    component: Optional[str] = typer.Option(None, "--component", help="Optional component filter."),
) -> None:
    """List collected profiler traces."""

    artifact_root = artifact_root.expanduser().resolve()
    traces = []
    for path in artifact_root.rglob("profile.json"):
        if component and component not in path.parts:
            continue
        traces.append(path)
    if not traces:
        console.print("[yellow]No profiler traces found.[/yellow]")
        return
    for trace in sorted(traces):
        console.print(trace)


def run_workflow(
    *,
    ingest_config: Optional[Path],
    build_datasets: bool,
    dataset_output_dir: Optional[Path],
    dataset_min_confidence: float,
    dataset_val_ratio: float,
    dataset_seed: int,
    embedding_config: Optional[Path],
    relation_config: Optional[Path],
    gnn_config: Optional[Path],
    export_config: Optional[Path],
    package_config: Optional[Path],
    promote_component: Optional[str],
    promote_version: Optional[str],
    manifest: Optional[Path],
    publish_url: Optional[str],
    artifact_root: Path,
    promote_publish: bool,
    force: bool,
    force_stage: List[str],
    force_lock: bool,
    mlflow_tracking_uri: Optional[str],
) -> Path:
    artifact_root = artifact_root.expanduser().resolve()
    state_path = artifact_root / "workflow-state.json"
    lock = WorkflowLock(artifact_root / "workflow.lock")
    try:
        lock.acquire(force=force_lock)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc))

    error: Optional[Exception] = None
    try:
        _workflow_impl(
            state_path=state_path,
            artifact_root=artifact_root,
            ingest_config=ingest_config,
            build_datasets=build_datasets,
            dataset_output_dir=dataset_output_dir,
            dataset_min_confidence=dataset_min_confidence,
            dataset_val_ratio=dataset_val_ratio,
            dataset_seed=dataset_seed,
            embedding_config=embedding_config,
            relation_config=relation_config,
            gnn_config=gnn_config,
            export_config=export_config,
            package_config=package_config,
            promote_component=promote_component,
            promote_version=promote_version,
            manifest=manifest,
            publish_url=publish_url,
            promote_publish=promote_publish,
            force=force,
            force_stage=force_stage,
        mlflow_tracking_uri=mlflow_tracking_uri,
        )
    except Exception as exc:  # pragma: no cover - surfaced in CLI tests
        error = exc
    finally:
        lock.release()
    report_path = generate_workflow_report(state_path, artifact_root, manifest)
    if error:
        raise error
    return report_path


def _workflow_impl(
    *,
    state_path: Path,
    artifact_root: Path,
    ingest_config: Optional[Path],
    build_datasets: bool,
    dataset_output_dir: Optional[Path],
    dataset_min_confidence: float,
    dataset_val_ratio: float,
    dataset_seed: int,
    embedding_config: Optional[Path],
    relation_config: Optional[Path],
    gnn_config: Optional[Path],
    export_config: Optional[Path],
    package_config: Optional[Path],
    promote_component: Optional[str],
    promote_version: Optional[str],
    manifest: Optional[Path],
    publish_url: Optional[str],
    promote_publish: bool,
    force: bool,
    force_stage: List[str],
    mlflow_tracking_uri: Optional[str],
) -> None:
    force_set = {stage.lower() for stage in force_stage}
    ingest_cfg = load_ingest_config(ingest_config) if ingest_config else None

    def _execute(stage_name: str, enabled: bool, output_path: Optional[Path], fn):
        state = load_workflow_state(state_path)
        if not _should_run_stage(state, stage_name, enabled, output_path, force, force_set):
            if enabled:
                console.print(f"[yellow]Skipping {stage_name} (up-to-date)[/yellow]")
            return
        record_stage_start(state_path, stage_name)
        try:
            fn()
        except Exception as exc:
            record_stage_failure(state_path, stage_name, exc)
            raise
        record_stage_success(state_path, stage_name, output_path)

    _execute(
        "ingest",
        ingest_cfg is not None,
        Path(ingest_cfg.output_dir) if ingest_cfg else None,
        lambda: IngestionTask(ingest_cfg).run() if ingest_cfg else None,
    )

    if build_datasets:
        if ingest_cfg is None:
            raise typer.BadParameter("--build-datasets requires --ingest-config to be set.")
        dataset_output = (dataset_output_dir or Path(ingest_cfg.output_dir) / "datasets").expanduser().resolve()

        def _build_datasets():
            result = build_datasets_from_ingestion(
                Path(ingest_cfg.output_dir) / "chunks.jsonl",
                Path(ingest_cfg.output_dir) / "relations.jsonl",
                dataset_output,
                val_ratio=dataset_val_ratio,
                min_confidence=dataset_min_confidence,
                seed=dataset_seed,
                parses_dir=Path(ingest_cfg.output_dir) / "cache" / "parses",
            )
            console.print(
                "[green]Datasets built at {dir}[/green] (embedding train={et}, val={ev}; "
                "relation train={rt}, val={rv}; gnn train={gt}, val={gv})".format(
                    dir=result.output_dir,
                    et=result.embedding_train,
                    ev=result.embedding_val,
                    rt=result.relation_train,
                    rv=result.relation_val,
                    gt=result.gnn_train,
                    gv=result.gnn_val,
                )
            )

        _execute("datasets", True, dataset_output, _build_datasets)

    if embedding_config:
        embedding_cfg, embedding_art_root = _resolve_component_config(embedding_config, "embedding")
        if mlflow_tracking_uri:
            embedding_cfg.mlflow_tracking_uri = embedding_cfg.mlflow_tracking_uri or mlflow_tracking_uri
        _execute("embedding", True, Path(embedding_cfg.output_dir), lambda: EmbeddingTask(embedding_cfg, embedding_art_root).run())

    if relation_config:
        relation_cfg, relation_art_root = _resolve_component_config(relation_config, "relation")
        if mlflow_tracking_uri:
            relation_cfg.mlflow_tracking_uri = relation_cfg.mlflow_tracking_uri or mlflow_tracking_uri
        _execute("relation", True, Path(relation_cfg.output_dir), lambda: RelationExtractionTask(relation_cfg, relation_art_root).run())

    if gnn_config:
        gnn_cfg, gnn_art_root = _resolve_component_config(gnn_config, "gnn")
        if mlflow_tracking_uri:
            gnn_cfg.mlflow_tracking_uri = gnn_cfg.mlflow_tracking_uri or mlflow_tracking_uri
        _execute("gnn", True, Path(gnn_cfg.output_dir), lambda: GNNTask(gnn_cfg, gnn_art_root).run())

    if export_config:
        export_cfg, export_art_root = _resolve_component_config(export_config, "export")
        manifest_path = Path(export_cfg.manifest_path)
        _execute("export", True, manifest_path, lambda: ExportTask(export_cfg, export_art_root).run())

    if package_config:
        package_cfg, package_art_root = _resolve_component_config(package_config, "export")
        package_output = Path(package_cfg.manifest_path).with_suffix(".tar.gz")
        _execute("package", True, package_output, lambda: PackageTask(package_cfg, package_art_root).run())

    if promote_component and promote_version:
        promote_output = artifact_root / promote_component / promote_version

        def _promote():
            metadata_path = promote_artifact(promote_component, promote_version, artifact_root)
            console.print(f"[green]Workflow promotion wrote {metadata_path}[/green]")
            if manifest:
                meta = show_artifact(promote_component, promote_version, artifact_root)
                _update_manifest(manifest, promote_component, meta)
                console.print(f"[blue]Manifest updated at {manifest}[/blue]")
                if publish_url or promote_publish:
                    target_url = publish_url or os.environ.get("SPHANA_ARTIFACT_PUBLISH_URL")
                    if not target_url:
                        raise typer.BadParameter("No publish URL provided via flag or SPHANA_ARTIFACT_PUBLISH_URL.")
                    publish_manifest(manifest, target_url)
                    console.print(f"[cyan]Manifest published to {target_url}[/cyan]")
            elif publish_url or promote_publish:
                raise typer.BadParameter("Publishing requires --manifest in workflow run.")

        _execute("promote", True, promote_output, _promote)



def _resolve_component_config(config_path: Path, attr: str):
    trainer_config: TrainerConfig = load_config(config_path)
    component = getattr(trainer_config, attr)
    if component is None:
        raise typer.BadParameter(f"No '{attr}' section found in {config_path}")
    artifact_root = trainer_config.artifact_root or trainer_config.workspace_dir
    return component, artifact_root


def _build_sweep_grid(options: Dict[str, List]) -> List[Dict[str, float]]:
    axes = [(key, values) for key, values in options.items() if values]
    if not axes:
        return [{}]
    keys = [key for key, _ in axes]
    value_lists = [values for _, values in axes]
    combos: List[Dict[str, float]] = []
    for values in itertools.product(*value_lists):
        combos.append(dict(zip(keys, values)))
    return combos


def _load_sweep_grid_file(path: Path) -> Dict[str, List]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Sweep grid file must contain a mapping. Got {type(payload)}")
    grid: Dict[str, List] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            normalized = []
            for item in value:
                if isinstance(item, str):
                    try:
                        item = float(item)
                    except ValueError:
                        pass
                normalized.append(item)
            grid[key] = normalized
        else:
            grid[key] = [value]
    return grid


def _record_sweep_result(component: str, artifact_root: Path, config, overrides: Dict, result) -> None:
    sweep_dir = Path(artifact_root) / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    log_path = sweep_dir / f"{component}.jsonl"
    metrics = getattr(result, "metrics", {}) or {}
    metric_keys = {"embedding": "val_cosine", "relation": "val_f1", "gnn": "val_loss"}
    score = metrics.get(metric_keys.get(component, ""))
    payload = {
        "component": component,
        "version": config.version,
        "overrides": overrides,
        "metrics": metrics,
        "score": score,
        "checkpoint_dir": str(getattr(result, "checkpoint_dir", "")),
        "onnx_path": str(getattr(result, "onnx_path", "")),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _generate_parity_bundle(artifact_root: Path, parity_dir: Path) -> List[Path]:
    from sphana_trainer.artifacts.parity import generate_parity_sample

    parity_dir = parity_dir.expanduser().resolve()
    parity_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for component, sample_file in PARITY_SAMPLE_FILES.items():
        if sample_file is None or not sample_file.exists():
            console.print(f"[yellow]Skipping {component} parity: no sample file configured[/yellow]")
            continue
        output = parity_dir / f"{component}-parity.json"
        report = generate_parity_sample(component, sample_file, artifact_root)
        output.write_text(json.dumps(report, indent=2))
        outputs.append(output)
    return outputs


def _summarize_metrics(metrics_path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"metrics": {}, "count": 0}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            summary["count"] += 1
            for key, value in (record.get("metrics") or {}).items():
                if not isinstance(value, (int, float)):
                    continue
                stats = summary["metrics"].setdefault(key, {"min": value, "max": value, "sum": 0.0})
                stats["min"] = min(stats["min"], value)
                stats["max"] = max(stats["max"], value)
                stats["sum"] += value
    for key, stats in summary["metrics"].items():
        stats["avg"] = stats["sum"] / max(1, summary["count"])
        del stats["sum"]
    return summary


def _update_manifest(manifest_path: Path, component: str, metadata) -> None:
    manifest_path = manifest_path.expanduser().resolve()
    payload = {"components": [], "artifacts": {}}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
    components = set(payload.get("components", []))
    components.add(component)
    payload["components"] = sorted(components)
    artifact_entry = metadata.quantized_path or metadata.onnx_path
    artifacts = payload.setdefault("artifacts", {})
    artifacts[component] = artifact_entry
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def _cache_relation_model(model_name: str) -> None:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise typer.BadParameter("transformers is required to cache relation models.") from exc

    console.print(f"[cyan]Downloading relation model {model_name}...[/cyan]")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)
    console.print(f"[green]Cached relation model {model_name}[/green]")


def _download_spacy_model(model_name: str) -> None:
    try:
        import spacy.cli
    except ImportError as exc:
        raise typer.BadParameter("spaCy is required to download spaCy models.") from exc
    console.print(f"[cyan]Downloading spaCy model {model_name}...[/cyan]")
    spacy.cli.download(model_name)
    console.print(f"[green]spaCy model {model_name} ready[/green]")


def _download_stanza(lang: str) -> None:
    try:
        import stanza
    except ImportError as exc:
        raise typer.BadParameter("Stanza is required to download Stanza pipelines.") from exc
    console.print(f"[cyan]Downloading Stanza resources for {lang}...[/cyan]")
    stanza.download(lang)
    console.print(f"[green]Stanza resources for {lang} ready[/green]")


if __name__ == "__main__":  # pragma: no cover
    app()


