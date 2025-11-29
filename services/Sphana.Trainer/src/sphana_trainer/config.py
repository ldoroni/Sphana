"""Configuration models and helpers for the trainer CLI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Literal, List

import yaml
from pydantic import BaseModel, Field, field_validator


class BaseComponentConfig(BaseModel):
    """Common fields shared by all trainable components."""

    model_name: str = Field(..., description="Identifier of the base model to finetune.")
    output_dir: Path = Field(default=Path("target/artifacts"), description="Where to store checkpoints.")
    dataset_path: Path = Field(default=Path("target/datasets"), description="Path to the prepared dataset (file or directory).")
    validation_path: Optional[Path] = Field(
        default=None, description="Optional explicit validation dataset path."
    )
    train_file: Optional[Path] = Field(default=None, description="Override for train split file.")
    validation_file: Optional[Path] = Field(default=None, description="Override for validation split file.")
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    epochs: int = Field(default=3, ge=0) # Updated ge=0 to allow export-only (0 epochs)
    seed: int = Field(default=42, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    gradient_accumulation: int = Field(default=1, ge=1)
    version: Optional[str] = Field(default=None, description="Optional semantic version for the artifact.")
    ddp: bool = Field(default=False, description="Enable torch.distributed training.")
    precision: Literal["fp32", "bf16", "fp16"] = Field(default="fp32")
    resume_from: Optional[Path] = Field(default=None, description="Checkpoint directory to resume from.")
    max_checkpoints: int = Field(default=5, ge=1, description="Number of historical checkpoints to retain.")
    profile_steps: int = Field(
        default=0,
        ge=0,
        description="Number of training steps to capture with the PyTorch profiler (0 disables profiling).",
    )
    log_to_mlflow: bool = Field(default=False, description="Enable MLflow logging for this component.")
    mlflow_tracking_uri: Optional[str] = Field(
        default=None, description="Optional MLflow tracking URI override."
    )
    mlflow_experiment: str = Field(default="sphana_trainer", description="MLflow experiment name.")
    mlflow_run_name: Optional[str] = Field(default=None, description="Optional MLflow run name override.")
    quantization_mismatch_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Allowed fraction of mismatched elements during quantization validation (e.g. 0.05 for 5%)."
    )
    @field_validator(
        "output_dir",
        "dataset_path",
        "validation_path",
        "train_file",
        "validation_file",
        "resume_from",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, value: Any) -> Optional[Path]:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        return Path(str(value)).expanduser().resolve()


class EmbeddingConfig(BaseComponentConfig):
    """Configuration for the embedding model."""

    max_seq_length: int = Field(default=512, ge=32)
    pooling_strategy: str = Field(default="mean")
    export_opset: int = Field(default=17, ge=13, le=19)
    quantize: bool = Field(default=True)
    temperature: float = Field(default=0.05, gt=0)
    eval_subset_size: Optional[int] = Field(
        default=None, description="Optional cap on validation samples for quick metrics."
    )
    metric_threshold: Optional[float] = Field(
        default=None, description="Minimum acceptable validation cosine similarity."
    )
    progress_log_interval: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Percentage interval for progress logging (1-100). Lower = more frequent logs."
    )


class RelationExtractionConfig(BaseComponentConfig):
    """Configuration for the relation extraction model."""

    dependency_cache: Path = Field(
        default=Path("target/cache/dep"), description="Location of cached dependency parses."
    )
    negative_sampling_ratio: float = Field(default=0.5, ge=0, le=1)
    export_opset: int = Field(default=17, ge=13, le=19)
    quantize: bool = Field(default=True)
    max_seq_length: int = Field(default=256, ge=64)
    weight_decay: float = Field(default=0.01, ge=0.0)
    eval_batch_size: int = Field(default=32, ge=1)
    early_stopping_patience: int = Field(default=2, ge=1)
    metric_threshold: Optional[float] = Field(
        default=None, description="Minimum acceptable macro F1 on the validation set."
    )
    progress_log_interval: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Percentage interval for progress logging (1-100). Lower = more frequent logs."
    )

    @field_validator("dependency_cache", mode="before")
    @classmethod
    def _expand_cache(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value)).expanduser().resolve()


class GNNConfig(BaseComponentConfig):
    """Configuration for the GNN reasoner."""

    hidden_dim: int = Field(default=256, ge=32)
    num_layers: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0, le=1)
    listwise_loss: str = Field(default="listnet")
    export_opset: int = Field(default=17, ge=13, le=19)
    quantize: bool = Field(default=True)
    max_nodes: int = Field(default=128, ge=8)
    max_edges: int = Field(default=512, ge=16)
    temperature: float = Field(default=1.0, gt=0)
    metric_threshold: Optional[float] = Field(
        default=None, description="Maximum acceptable validation loss (lower is better)."
    )
    progress_log_interval: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Percentage interval for progress logging (1-100). Lower = more frequent logs."
    )


class NerConfig(BaseComponentConfig):
    """Configuration for the NER model."""
    
    export_opset: int = Field(default=17, ge=13, le=19)
    quantize: bool = Field(default=True)
    max_seq_length: int = Field(default=512, ge=32)


class LlmConfig(BaseComponentConfig):
    """Configuration for the LLM generator."""
    
    export_opset: int = Field(default=17, ge=13, le=19)
    quantize: bool = Field(default=True)
    max_seq_length: int = Field(default=512, ge=32)


class ExportConfig(BaseModel):
    """Configuration for packaging/exporting ONNX artifacts."""

    manifest_path: Path = Field(default=Path("target/manifests/latest.json"))
    include_models: list[str] = Field(default_factory=lambda: ["embedding", "relation", "gnn", "ner", "llm"])
    publish_uri: Optional[str] = Field(default=None, description="Optional remote artifact store URI.")
    artifact_root: Path = Field(default=Path("target/artifacts"))

    @field_validator("manifest_path", mode="before")
    @classmethod
    def _expand_manifest(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value)).expanduser().resolve()


class TrainerConfig(BaseModel):
    """Root configuration for the trainer application."""

    workspace_dir: Path = Field(default=Path("target/workspace"))
    tracking_uri: Optional[str] = Field(default=None)
    artifact_root: Path = Field(default=Path("target/artifacts"))
    embedding: Optional[EmbeddingConfig] = None
    relation: Optional[RelationExtractionConfig] = None
    gnn: Optional[GNNConfig] = None
    ner: Optional[NerConfig] = None
    llm: Optional[LlmConfig] = None
    export: Optional[ExportConfig] = None

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def _expand_workspace(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value)).expanduser().resolve()


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=32)
def load_config(path: Optional[Path]) -> TrainerConfig:
    """Load a TrainerConfig from a YAML file."""

    if path is None:
        return TrainerConfig()

    resolved = path.expanduser().resolve()
    data = _read_yaml(resolved)
    return TrainerConfig.model_validate(data)


def dump_config(config: TrainerConfig, path: Path) -> None:
    """Persist a TrainerConfig to disk."""

    payload = config.model_dump(mode="json", by_alias=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


class IngestionConfig(BaseModel):
    source: str = Field(
        description="Path or glob pattern to input files. "
        "Examples: 'file.jsonl', 'file.jsonl.gz', 'dir/*.jsonl.gz', 'dir/**/*.txt'"
    )
    chunks_output_dir: Path = Field(
        description="Directory for chunks output files"
    )
    relations_output_dir: Path = Field(
        description="Directory for relations output files"
    )
    output_compressed: bool = Field(
        default=False,
        description="Compress output files (chunks.jsonl.gz, relations.jsonl.gz) with gzip"
    )
    cache_dir: Optional[Path] = Field(default=None)
    cache_enabled: bool = Field(default=True)
    chunk_size: int = Field(default=120, ge=10)
    chunk_overlap: int = Field(default=20, ge=0)
    parser: Literal["simple", "spacy", "stanza"] = Field(
        default="simple", description="Relation extraction backend to use."
    )
    parser_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model name or custom pipeline identifier (used for parser='spacy').",
    )
    language: str = Field(default="en", description="Language code for parser='stanza'.")
    relation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    relation_model: Optional[str] = Field(default=None, description="Optional Hugging Face model for scoring relations.")
    relation_max_length: int = Field(default=256, ge=32)
    relation_calibration: Optional[Path] = Field(
        default=None, description="Optional JSON file with label calibration coefficients."
    )
    progress_log_interval: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Percentage interval for progress logging (1-100). Lower = more frequent logs."
    )

    @field_validator("chunks_output_dir", "relations_output_dir", "cache_dir", mode="before")
    @classmethod
    def _expand_ingest_paths(cls, value: Any) -> Optional[Path]:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        return Path(str(value)).expanduser().resolve()


class DatasetBuildConfig(BaseModel):
    """Configuration for dataset building from ingestion outputs."""
    
    chunks_pattern: str = Field(
        description="Glob pattern for chunks files (e.g., 'target/ingest/chunks/*.jsonl')"
    )
    relations_pattern: str = Field(
        description="Glob pattern for relations files (e.g., 'target/ingest/relations/*.jsonl')"
    )
    output_dir: Path = Field(
        default=Path("target/datasets"),
        description="Directory to write derived datasets"
    )
    min_confidence: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum relation confidence to keep"
    )
    val_ratio: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Validation split ratio"
    )
    seed: int = Field(
        default=42,
        description="Random seed for deterministic shuffling"
    )
    extra_embedding: List[Path] = Field(
        default_factory=list,
        description="Additional embedding JSONL files"
    )
    extra_relation: List[Path] = Field(
        default_factory=list,
        description="Additional relation JSONL files"
    )
    extra_gnn: List[Path] = Field(
        default_factory=list,
        description="Additional GNN JSONL files"
    )
    parses_dir: Optional[Path] = Field(
        default=None,
        description="Optional directory containing cached parse JSON files"
    )
    output_compressed: bool = Field(
        default=False,
        description="Compress output datasets with gzip (.jsonl.gz)"
    )
    
    @field_validator("output_dir", "parses_dir", mode="before")
    @classmethod
    def _expand_paths(cls, value: Any) -> Optional[Path]:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        return Path(str(value)).expanduser().resolve()


def load_ingest_config(path: Path) -> IngestionConfig:
    resolved = path.expanduser().resolve()
    data = _read_yaml(resolved)
    if "ingest" in data:
        return IngestionConfig.model_validate(data["ingest"])
    return IngestionConfig.model_validate(data)


def load_dataset_build_config(path: Path) -> DatasetBuildConfig:
    resolved = path.expanduser().resolve()
    data = _read_yaml(resolved)
    if "dataset_build" in data:
        return DatasetBuildConfig.model_validate(data["dataset_build"])
    return DatasetBuildConfig.model_validate(data)
