"""Tests for configuration helpers."""

from pathlib import Path

import yaml

from sphana_trainer.config import (
    EmbeddingConfig,
    RelationExtractionConfig,
    ExportConfig,
    TrainerConfig,
    IngestionConfig,
    load_config,
    dump_config,
    load_ingest_config,
)


def test_path_expansion_and_defaults(tmp_path, monkeypatch):
    cfg = EmbeddingConfig(
        model_name="model",
        dataset_path=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        train_file=str(tmp_path / "train.jsonl"),
        validation_file=None,
    )
    assert cfg.dataset_path == (tmp_path / "data").resolve()
    assert cfg.validation_file is None

    rel_cfg = RelationExtractionConfig(
        model_name="model",
        dataset_path=str(tmp_path / "data"),
        dependency_cache=str(tmp_path / "cache"),
    )
    assert rel_cfg.dependency_cache == (tmp_path / "cache").resolve()

    # Path inputs remain unchanged
    path_obj = tmp_path / "direct"
    cfg_path = EmbeddingConfig(model_name="model", dataset_path=path_obj)
    assert cfg_path.dataset_path == path_obj

    rel_cfg_path = RelationExtractionConfig(model_name="model", dataset_path=path_obj, dependency_cache=path_obj)
    assert rel_cfg_path.dependency_cache == path_obj

    export_cfg = ExportConfig(manifest_path=str(tmp_path / "manifest.json"))
    assert export_cfg.manifest_path == (tmp_path / "manifest.json").resolve()

    export_cfg_path = ExportConfig(manifest_path=path_obj)
    assert export_cfg_path.manifest_path == path_obj

    trainer_cfg = TrainerConfig(workspace_dir=str(tmp_path / "workspace"))
    assert trainer_cfg.workspace_dir == (tmp_path / "workspace").resolve()

    trainer_cfg_path = TrainerConfig(workspace_dir=path_obj)
    assert trainer_cfg_path.workspace_dir == path_obj


def test_load_and_dump_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    data = {
        "artifact_root": str(tmp_path / "art"),
        "embedding": {
            "model_name": "dummy",
            "dataset_path": str(tmp_path / "data"),
        },
    }
    cfg_path.write_text(yaml.safe_dump(data))

    loaded = load_config(cfg_path)
    assert loaded.embedding.model_name == "dummy"

    output = tmp_path / "roundtrip.yaml"
    dump_config(loaded, output)
    reloaded = yaml.safe_load(output.read_text())
    assert reloaded["artifact_root"].endswith("art")


def test_load_config_none_returns_defaults():
    cfg = load_config(None)
    assert isinstance(cfg, TrainerConfig)
    assert cfg.embedding is None


def test_ingestion_config_paths(tmp_path):
    cfg = IngestionConfig(
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        cache_dir=None,
    )
    assert cfg.input_dir == (tmp_path / "in").resolve()
    assert cfg.cache_dir is None
    assert cfg.parser == "simple"
    assert cfg.parser_model == "en_core_web_sm"

    cfg_custom = IngestionConfig(
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        parser="stanza",
        language="de",
    )
    assert cfg_custom.parser == "stanza"
    assert cfg_custom.language == "de"


def test_load_ingest_config_with_nested_key(tmp_path):
    cfg_path = tmp_path / "ingest.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"ingest": {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}}
        )
    )
    cfg = load_ingest_config(cfg_path)
    assert isinstance(cfg, IngestionConfig)

    cfg_path.write_text(yaml.safe_dump({"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}))
    cfg = load_ingest_config(cfg_path)
    assert isinstance(cfg, IngestionConfig)

