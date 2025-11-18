"""Basic sanity tests for the CLI."""

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import typer
import yaml
from typer.testing import CliRunner

from sphana_trainer.cli import (
    app,
    main,
    _build_sweep_grid,
    _cache_relation_model,
    _download_spacy_model,
    _download_stanza,
    _fetch_wiki_summary,
    _generate_parity_bundle,
    _load_sweep_grid_file,
    _should_run_stage,
    _summarize_metrics,
    _workflow_impl,
    run_workflow,
)
from sphana_trainer.data.pipeline import IngestionResult  # type: ignore[import]

TEST_DATA_ROOT = (Path(__file__).resolve().parents[1] / "data").resolve()
SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "sphana_trainer" / "schemas"
EMBED_DATA = TEST_DATA_ROOT / "embedding"
RELATION_DATA = TEST_DATA_ROOT / "relation"
GRAPHS_DATA = TEST_DATA_ROOT / "graphs"
INGEST_SAMPLE = TEST_DATA_ROOT / "ingest_sample"


runner = CliRunner()


def test_help_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Sphana neural training CLI" in result.stdout


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "sphana-trainer" in result.stdout.lower()


def test_cli_without_subcommand_prints_help():
    result = runner.invoke(app, [], standalone_mode=False)
    assert "Sphana neural training CLI" in result.stdout


def test_cli_missing_component_section(tmp_path):
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text(yaml.safe_dump({"workspace_dir": str(tmp_path)}))

    result = runner.invoke(app, ["train", "embedding", "--config", str(cfg_path)])
    assert result.exit_code != 0
    assert "No 'embedding' section" in (result.stderr or result.stdout)


def test_cli_callback_main_exits_with_help():
    class DummyCtx:
        invoked_subcommand = None

        def get_help(self):
            return "HELP"

    ctx = DummyCtx()
    with pytest.raises(typer.Exit) as exc:
        main(ctx)
    assert exc.value.exit_code == 0


def test_ingest_command(monkeypatch, tmp_path):
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"ingest": {"input_dir": str(INGEST_SAMPLE), "output_dir": str(tmp_path / "out")}}
        )
    )

    called = {}

    class FakeTask:
        def __init__(self, config):
            called["config"] = config

        def run(self):
            called["ran"] = True

    monkeypatch.setattr("sphana_trainer.cli.IngestionTask", FakeTask)
    result = runner.invoke(app, ["ingest", "--config", str(cfg)])
    assert result.exit_code == 0
    assert called.get("ran") is True


def test_ingest_validate_command(monkeypatch, tmp_path):
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text(f"input_dir: {INGEST_SAMPLE}\n")

    chunks = tmp_path / "chunks.jsonl"
    relations = tmp_path / "relations.jsonl"
    chunks.write_text('{"id": "c1", "document_id": "d1", "text": "body"}\n')
    relations.write_text("")

    fake_result = IngestionResult(
        chunks_path=chunks,
        relations_path=relations,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        document_count=1,
        chunk_count=1,
        relation_count=0,
    )

    class FakePipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, force=False):
            assert force is False
            return fake_result

    monkeypatch.setattr("sphana_trainer.cli.IngestionPipeline", FakePipeline)
    chunk_schema = SCHEMA_ROOT / "ingestion" / "chunks.schema.json"
    relation_schema = SCHEMA_ROOT / "ingestion" / "relations.schema.json"
    result = runner.invoke(
        app,
        [
            "ingest-validate",
            "--config",
            str(cfg),
            "--stats",
            "--chunks-schema",
            str(chunk_schema),
            "--relations-schema",
            str(relation_schema),
        ],
    )
    assert result.exit_code == 0
    assert "Ingestion valid" in result.stdout
    assert "Labels" in result.stdout


def test_workflow_run_delegates_to_helper(monkeypatch, tmp_path):
    captured = {}

    def fake_run_workflow(**kwargs):
        captured.update(kwargs)
        return tmp_path / "report.json"

    monkeypatch.setattr("sphana_trainer.cli.run_workflow", fake_run_workflow)
    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "--artifact-root",
            str(tmp_path),
            "--force-lock",
        ],
    )
    assert result.exit_code == 0
    assert captured["artifact_root"] == Path(str(tmp_path))
    assert captured["force_lock"] is True


def test_workflow_run_lock_error(monkeypatch, tmp_path):
    def fake_acquire(self, *, force: bool = False):
        raise RuntimeError("locked")

    monkeypatch.setattr("sphana_trainer.cli.WorkflowLock.acquire", fake_acquire)
    monkeypatch.setattr("sphana_trainer.cli.WorkflowLock.release", lambda self: None)
    result = runner.invoke(app, ["workflow", "run", "--artifact-root", str(tmp_path)])
    assert result.exit_code != 0


def test_run_workflow_executes_stages(monkeypatch, tmp_path):
    calls = []
    ingest_cfg = SimpleNamespace(output_dir=str(tmp_path / "ingest"), cache_dir=str(tmp_path / "ingest" / "cache"))
    monkeypatch.setattr("sphana_trainer.cli.load_ingest_config", lambda path: ingest_cfg)

    class DummyTask:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self):
            calls.append("ingest")

    monkeypatch.setattr("sphana_trainer.cli.IngestionTask", DummyTask)

    def fake_build(chunks, relations, output, **kwargs):
        calls.append("datasets")
        output.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            embedding_train=1,
            embedding_val=0,
            relation_train=1,
            relation_val=0,
            gnn_train=1,
            gnn_val=0,
            output_dir=output,
        )

    monkeypatch.setattr("sphana_trainer.cli.build_datasets_from_ingestion", fake_build)

    class FakeTrainTask:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self):
            calls.append("train")
            return SimpleNamespace(checkpoint_dir=tmp_path, onnx_path=tmp_path / "model.onnx", metrics={"val": 1.0})

    monkeypatch.setattr("sphana_trainer.cli.EmbeddingTask", FakeTrainTask)
    monkeypatch.setattr("sphana_trainer.cli.RelationExtractionTask", FakeTrainTask)
    monkeypatch.setattr("sphana_trainer.cli.GNNTask", FakeTrainTask)

    class FakeExport:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self):
            calls.append("export")

    class FakePackage(FakeExport):
        def run(self):
            calls.append("package")

    monkeypatch.setattr("sphana_trainer.cli.ExportTask", FakeExport)
    monkeypatch.setattr("sphana_trainer.cli.PackageTask", FakePackage)
    monkeypatch.setattr("sphana_trainer.cli.promote_artifact", lambda *a, **k: tmp_path / "meta.json")
    monkeypatch.setattr("sphana_trainer.cli._update_manifest", lambda *a, **k: calls.append("manifest"))
    monkeypatch.setattr("sphana_trainer.cli.publish_manifest", lambda *a, **k: calls.append("publish"))
    monkeypatch.setattr("sphana_trainer.cli.show_artifact", lambda *a, **k: SimpleNamespace(version="v1", quantized_path="", onnx_path="model"))

    def fake_resolve(path, attr):
        data = {"output_dir": str(tmp_path / attr)}
        if attr == "export":
            data["manifest_path"] = str(tmp_path / f"{attr}.json")
        return SimpleNamespace(**data), tmp_path

    monkeypatch.setattr("sphana_trainer.cli._resolve_component_config", fake_resolve)

    report = run_workflow(
        ingest_config=tmp_path / "ingest.yaml",
        build_datasets=True,
        dataset_output_dir=tmp_path / "datasets",
        dataset_min_confidence=0.1,
        dataset_val_ratio=0.5,
        dataset_seed=0,
        embedding_config=tmp_path / "emb.yaml",
        relation_config=tmp_path / "rel.yaml",
        gnn_config=tmp_path / "gnn.yaml",
        export_config=tmp_path / "export.yaml",
        package_config=tmp_path / "package.yaml",
        promote_component="embedding",
        promote_version="v1",
        manifest=tmp_path / "manifest.json",
        publish_url="http://example",
        artifact_root=tmp_path / "artifacts",
        promote_publish=True,
        force=True,
        force_stage=[],
        force_lock=True,
        mlflow_tracking_uri=None,
    )
    assert report.exists()
    assert "ingest" in calls
    assert "datasets" in calls


def test_workflow_impl_skips_completed_stage(monkeypatch, tmp_path, capsys):
    state_path = tmp_path / "workflow-state.json"
    embed_dir = tmp_path / "embedding"
    embed_dir.mkdir()
    state_path.write_text(json.dumps({"stages": {"embedding": {"status": "succeeded", "output": str(embed_dir)}}}))

    def fake_resolve(path, attr):
        return (SimpleNamespace(output_dir=str(embed_dir)), tmp_path)

    class ExplodingTask:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("should not instantiate")

    monkeypatch.setattr("sphana_trainer.cli._resolve_component_config", fake_resolve)
    monkeypatch.setattr("sphana_trainer.cli.EmbeddingTask", ExplodingTask)
    _workflow_impl(
        state_path=state_path,
        artifact_root=tmp_path,
        ingest_config=None,
        build_datasets=False,
        dataset_output_dir=None,
        dataset_min_confidence=0.2,
        dataset_val_ratio=0.2,
        dataset_seed=42,
        embedding_config=tmp_path / "emb.yaml",
        relation_config=None,
        gnn_config=None,
        export_config=None,
        package_config=None,
        promote_component=None,
        promote_version=None,
        manifest=None,
        publish_url=None,
        promote_publish=False,
        force=False,
        force_stage=[],
        mlflow_tracking_uri=None,
    )
    captured = capsys.readouterr()
    assert "Skipping embedding" in captured.out


def test_workflow_wiki_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run_workflow(**kwargs):
        captured.update(kwargs)
        return tmp_path / "report.json"

    monkeypatch.setattr("sphana_trainer.cli.run_workflow", fake_run_workflow)
    monkeypatch.setattr("sphana_trainer.cli._generate_parity_bundle", lambda *a, **k: [tmp_path / "parity.json"])
    result = runner.invoke(
        app,
        [
            "workflow",
            "wiki",
            "--artifact-root",
            str(tmp_path),
            "--no-parity",
        ],
    )
    assert result.exit_code == 0
    assert captured["build_datasets"] is True


def test_workflow_wiki_generates_parity(monkeypatch, tmp_path):
    monkeypatch.setattr("sphana_trainer.cli.run_workflow", lambda **kwargs: tmp_path / "report.json")
    parity_outputs = [tmp_path / "parity.json"]
    monkeypatch.setattr("sphana_trainer.cli._generate_parity_bundle", lambda *a, **k: parity_outputs)
    result = runner.invoke(app, ["workflow", "wiki", "--artifact-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "Parity fixtures written" in result.stdout


def test_workflow_status(monkeypatch, tmp_path):
    state_file = tmp_path / "artifacts" / "workflow-state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps({"stages": {"embedding": {"timestamp": "now", "output": str(tmp_path / "embedding")}}})
    )
    result = runner.invoke(app, ["workflow", "status", "--artifact-root", str(tmp_path / "artifacts")])
    assert result.exit_code == 0
    assert "embedding" in result.stdout


def test_workflow_status_shows_errors(tmp_path):
    state_file = tmp_path / "artifacts" / "workflow-state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps({"stages": {"embedding": {"status": "failed", "output": "-", "error": "boom"}}})
    )
    result = runner.invoke(app, ["workflow", "status", "--artifact-root", str(tmp_path / "artifacts")])
    assert result.exit_code == 0
    assert "error=boom" in result.stdout


def test_workflow_status_no_state(tmp_path):
    result = runner.invoke(app, ["workflow", "status", "--artifact-root", str(tmp_path / "artifacts")])
    assert result.exit_code == 0


def test_workflow_run_requires_ingest_for_datasets(tmp_path):
    result = runner.invoke(app, ["workflow", "run", "--build-datasets"])
    assert result.exit_code != 0


def test_workflow_run_publish_requires_manifest(monkeypatch, tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    meta_path = artifact_root / "embedding" / "v1" / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("{}")

    monkeypatch.setattr("sphana_trainer.cli.promote_artifact", lambda *a, **k: meta_path)
    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "--promote-component",
            "embedding",
            "--promote-version",
            "v1",
            "--artifact-root",
            str(artifact_root),
            "--publish",
        ],
    )
    assert result.exit_code != 0


def test_workflow_run_publish_requires_url(monkeypatch, tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    meta_path = artifact_root / "embedding" / "v1" / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr("sphana_trainer.cli.promote_artifact", lambda *a, **k: meta_path)
    class DummyMeta(SimpleNamespace):
        version: str = "v1"
        onnx_path: str = "model.onnx"
        quantized_path: str = ""
        metrics: dict = {}
        checkpoint_dir: str = str(tmp_path)
    monkeypatch.setattr("sphana_trainer.cli.show_artifact", lambda *a, **k: DummyMeta())
    monkeypatch.setattr("sphana_trainer.cli._update_manifest", lambda *a, **k: None)
    monkeypatch.setattr("sphana_trainer.cli.publish_manifest", lambda *a, **k: None)
    monkeypatch.delenv("SPHANA_ARTIFACT_PUBLISH_URL", raising=False)
    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "--promote-component",
            "embedding",
            "--promote-version",
            "v1",
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(artifact_root),
            "--publish",
        ],
    )
    assert result.exit_code != 0


def test_ingest_cache_models(monkeypatch):
    called = {}
    monkeypatch.setattr("sphana_trainer.cli._cache_relation_model", lambda name: called.setdefault("relation", name))
    monkeypatch.setattr("sphana_trainer.cli._download_spacy_model", lambda name: called.setdefault("spacy", name))
    monkeypatch.setattr("sphana_trainer.cli._download_stanza", lambda name: called.setdefault("stanza", name))
    result = runner.invoke(
        app,
        [
            "ingest-cache-models",
            "--relation-model",
            "model",
            "--spacy-model",
            "en_core_web_sm",
            "--stanza-lang",
            "en",
        ],
    )
    assert result.exit_code == 0
    assert called["relation"] == "model"
    assert called["spacy"] == "en_core_web_sm"
    assert called["stanza"] == "en"


def test_dataset_download_wiki(monkeypatch, tmp_path):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"pageid": 1, "title": "Test", "extract": "Some text"}

    class FakeSession:
        def get(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr("sphana_trainer.cli.requests.Session", lambda: FakeSession())
    output = tmp_path / "wiki.jsonl"
    result = runner.invoke(app, ["dataset-download-wiki", "--output", str(output), "--title", "Test"])
    assert result.exit_code == 0
    assert output.exists()


def test_dataset_download_wiki_titles_file(monkeypatch, tmp_path):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"pageid": 2, "title": "FromFile", "extract": "File text"}

    class FakeSession:
        def get(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr("sphana_trainer.cli.requests.Session", lambda: FakeSession())
    titles_path = tmp_path / "titles.txt"
    titles_path.write_text("FromFile\n")
    output = tmp_path / "wiki.jsonl"
    result = runner.invoke(
        app,
        [
            "dataset-download-wiki",
            "--output",
            str(output),
            "--titles-file",
            str(titles_path),
            "--no-shuffle",
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert output.exists()
    assert "FromFile" in output.read_text()


def test_dataset_download_wiki_handles_failures(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.RequestException("boom")
            if self.calls == 2:
                resp = SimpleNamespace(status_code=500)
                resp.json = lambda: {}
                return resp
            resp = SimpleNamespace(status_code=200)
            resp.json = lambda: {"pageid": 1, "title": "Empty", "extract": ""}
            return resp

    monkeypatch.setattr("sphana_trainer.cli.requests.Session", lambda: FakeSession())
    result = runner.invoke(
        app,
        ["dataset-download-wiki", "--output", str(tmp_path / "wiki.jsonl"), "--title", "A", "--title", "B", "--title", "C"],
    )
    assert result.exit_code != 0


def test_fetch_wiki_summary_retries_and_raises():
    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            raise requests.RequestException("boom")

    with pytest.raises(requests.RequestException):
        _fetch_wiki_summary(Session(), "Retry", attempts=2)


def test_fetch_wiki_summary_returns_none_on_http_error():
    class Response:
        status_code = 500

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    assert _fetch_wiki_summary(Session(), "Fail", attempts=1) is None


def test_fetch_wiki_summary_returns_none_on_empty_extract():
    class Response:
        status_code = 200

        def json(self):
            return {"title": "Empty", "extract": ""}

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    assert _fetch_wiki_summary(Session(), "Empty", attempts=1) is None


def test_fetch_wiki_summary_zero_attempts():
    class Session:
        def get(self, *_args, **_kwargs):
            raise AssertionError("should not be called")

    assert _fetch_wiki_summary(Session(), "Zero", attempts=0) is None


def test_dataset_download_wiki_handles_fetch_exception(monkeypatch, tmp_path):
    output = tmp_path / "wiki.jsonl"

    def fake_fetch(session, title):
        if title == "Bad":
            raise requests.RequestException("boom")
        return {"id": 1, "title": "Good", "text": "body", "source": "wikipedia"}

    class DummySession:
        def __init__(self):
            self.headers = {}

    monkeypatch.setattr("sphana_trainer.cli._fetch_wiki_summary", fake_fetch)
    monkeypatch.setattr("sphana_trainer.cli.requests.Session", lambda: DummySession())
    result = runner.invoke(
        app,
        [
            "dataset-download-wiki",
            "--output",
            str(output),
            "--title",
            "Bad",
            "--title",
            "Good",
        ],
    )
    assert result.exit_code == 0
    assert "Good" in output.read_text()


def test_dataset_download_wiki_retries_success(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self):
            self.headers = {}
            self.calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.RequestException("boom")
            resp = SimpleNamespace(status_code=200)
            resp.json = lambda: {"pageid": 1, "title": "Retry", "extract": "content"}
            return resp

    monkeypatch.setattr("sphana_trainer.cli.requests.Session", lambda: FakeSession())
    output = tmp_path / "wiki.jsonl"
    result = runner.invoke(app, ["dataset-download-wiki", "--output", str(output), "--title", "Retry"])
    assert result.exit_code == 0
    assert "Retry" in output.read_text()


def test_metrics_summarize_command(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = run_dir / "metrics.jsonl"
    metrics.write_text('{"metrics": {"val": 0.5}}\n{"metrics": {"val": 1.0}}\n')
    result = runner.invoke(app, ["metrics", "summarize", str(run_dir)])
    assert result.exit_code == 0
    assert '"count": 2' in result.stdout


def test_metrics_summarize_missing_file(tmp_path):
    result = runner.invoke(app, ["metrics", "summarize", str(tmp_path / "missing.jsonl")])
    assert result.exit_code != 0


def test_metrics_summarize_missing_file_in_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = runner.invoke(app, ["metrics", "summarize", str(run_dir)])
    assert result.exit_code != 0


def test_summarize_metrics_skips_blank_and_strings(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('\n{"metrics": {"val": "n/a"}}\n{"metrics": {"val": 1.0}}\n')
    summary = _summarize_metrics(metrics_path)
    assert summary["metrics"]["val"]["min"] == 1.0
    assert summary["metrics"]["val"]["max"] == 1.0


def test_profile_traces_command(tmp_path):
    trace = tmp_path / "embedding" / "run" / "profile.json"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}")
    result = runner.invoke(app, ["profile", "traces", "--artifact-root", str(tmp_path), "--component", "embedding"])
    assert result.exit_code == 0
    assert "profile.json" in result.stdout


def test_profile_traces_no_matches(tmp_path):
    result = runner.invoke(app, ["profile", "traces", "--artifact-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "No profiler traces found" in result.stdout


def test_profile_traces_component_filter_skips(tmp_path):
    trace = tmp_path / "relation" / "run" / "profile.json"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}")
    result = runner.invoke(app, ["profile", "traces", "--artifact-root", str(tmp_path), "--component", "embedding"])
    assert result.exit_code == 0
    assert "No profiler traces found" in result.stdout


def test_dataset_validate_requires_schema_or_type(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"query": "q", "positive": "p"}\n')
    result = runner.invoke(app, ["dataset-validate", str(data)])
    assert result.exit_code != 0
    assert "type" in (result.stderr or result.stdout)


def test_dataset_validate_unknown_type(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"query": "q", "positive": "p"}\n')
    result = runner.invoke(app, ["dataset-validate", str(data), "--type", "unknown"])
    assert result.exit_code != 0
    assert "Unsupported dataset type" in (result.stderr or result.stdout)


def test_artifacts_parity_samples(monkeypatch, tmp_path):
    sample = tmp_path / "sample.jsonl"
    sample.write_text("{}\n")
    output = tmp_path / "parity.json"

    def fake_generate(component, sample_file, artifact_root):
        assert component == "embedding"
        assert sample_file == sample
        return {"component": component, "inputs": {"text": "demo"}, "outputs": {"embedding": [0.0]}}

    monkeypatch.setattr("sphana_trainer.artifacts.parity.generate_parity_sample", fake_generate)
    result = runner.invoke(
        app,
        [
            "artifacts",
            "parity-samples",
            "embedding",
            str(sample),
            str(output),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["component"] == "embedding"


def test_generate_parity_bundle(monkeypatch, tmp_path):
    sample = tmp_path / "sample.jsonl"
    sample.write_text("{}\n")
    monkeypatch.setattr("sphana_trainer.cli.PARITY_SAMPLE_FILES", {"embedding": sample})
    monkeypatch.setattr(
        "sphana_trainer.artifacts.parity.generate_parity_sample",
        lambda component, sample_file, artifact_root: {"component": component},
    )
    outputs = _generate_parity_bundle(tmp_path, tmp_path / "parity")
    assert outputs
    assert outputs[0].exists()


def test_generate_parity_bundle_skips_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("sphana_trainer.cli.PARITY_SAMPLE_FILES", {"embedding": tmp_path / "missing.jsonl"})
    outputs = _generate_parity_bundle(tmp_path, tmp_path / "parity")
    assert outputs == []


def test_artifacts_list_handles_empty(monkeypatch):
    monkeypatch.setattr("sphana_trainer.cli.list_artifacts", lambda *a, **k: {})
    result = runner.invoke(app, ["artifacts", "list"])
    assert result.exit_code == 0


def test_artifacts_list_component_with_no_entries(monkeypatch):
    monkeypatch.setattr("sphana_trainer.cli.list_artifacts", lambda *a, **k: {"embedding": []})
    result = runner.invoke(app, ["artifacts", "list"])
    assert result.exit_code == 0


def test_train_commands(monkeypatch, tmp_path):
    embedding_cfg = tmp_path / "emb.yaml"
    relation_cfg = tmp_path / "rel.yaml"
    gnn_cfg = tmp_path / "gnn.yaml"
    base = {
        "workspace_dir": str(tmp_path),
        "artifact_root": str(tmp_path / "artifacts"),
    }
    emb_payload = base | {
        "embedding": {
            "model_name": "stub",
            "dataset_path": str(EMBED_DATA),
            "train_file": str(EMBED_DATA / "train.jsonl"),
            "validation_file": str(EMBED_DATA / "validation.jsonl"),
            "output_dir": str(tmp_path / "emb"),
        }
    }
    rel_payload = base | {
        "relation": {
            "model_name": "stub",
            "dataset_path": str(RELATION_DATA),
            "train_file": str(RELATION_DATA / "train.jsonl"),
            "validation_file": str(RELATION_DATA / "validation.jsonl"),
            "output_dir": str(tmp_path / "rel"),
        }
    }
    gnn_payload = base | {
        "gnn": {
            "model_name": "stub",
            "dataset_path": str(GRAPHS_DATA),
            "train_file": str(GRAPHS_DATA / "train.jsonl"),
            "validation_file": str(GRAPHS_DATA / "validation.jsonl"),
            "output_dir": str(tmp_path / "gnn"),
        }
    }
    embedding_cfg.write_text(yaml.safe_dump(emb_payload))
    relation_cfg.write_text(yaml.safe_dump(rel_payload))
    gnn_cfg.write_text(yaml.safe_dump(gnn_payload))

    called = {"embedding": False, "relation": False, "gnn": False}

    class FakeTask:
        def __init__(self, cfg, *_args, **_kwargs):
            self.cfg = cfg

        def run(self):
            called["current"] = self.cfg

    monkeypatch.setattr("sphana_trainer.cli.EmbeddingTask", FakeTask)
    result = runner.invoke(
        app,
        [
            "train",
            "embedding",
            "--config",
            str(embedding_cfg),
            "--precision",
            "fp16",
            "--grad-accum",
            "2",
            "--profile-steps",
            "5",
        ],
    )
    assert result.exit_code == 0
    assert called["current"].precision == "fp16"
    assert called["current"].gradient_accumulation == 2
    assert called["current"].profile_steps == 5

    monkeypatch.setattr("sphana_trainer.cli.RelationExtractionTask", FakeTask)
    result = runner.invoke(
        app,
        ["train", "relation", "--config", str(relation_cfg), "--precision", "bf16", "--grad-accum", "3"],
    )
    assert result.exit_code == 0

    monkeypatch.setattr("sphana_trainer.cli.GNNTask", FakeTask)
    result = runner.invoke(
        app,
        ["train", "gnn", "--config", str(gnn_cfg), "--precision", "fp16", "--grad-accum", "4"],
    )
    assert result.exit_code == 0


def test_train_relation_profile_steps(monkeypatch, tmp_path):
    cfg_path = tmp_path / "rel.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": str(tmp_path),
                "artifact_root": str(tmp_path / "artifacts"),
                "relation": {
                    "model_name": "stub",
                    "dataset_path": str(RELATION_DATA),
                    "train_file": str(RELATION_DATA / "train.jsonl"),
                    "validation_file": str(RELATION_DATA / "validation.jsonl"),
                    "output_dir": str(tmp_path / "rel"),
                },
            }
        )
    )
    captured = {}

    class FakeTask:
        def __init__(self, cfg, *_args, **_kwargs):
            captured["config"] = cfg

        def run(self):
            pass

    monkeypatch.setattr("sphana_trainer.cli.RelationExtractionTask", FakeTask)
    result = runner.invoke(
        app,
        ["train", "relation", "--config", str(cfg_path), "--profile-steps", "7"],
    )
    assert result.exit_code == 0
    assert captured["config"].profile_steps == 7


def test_train_gnn_profile_steps(monkeypatch, tmp_path):
    cfg_path = tmp_path / "gnn.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": str(tmp_path),
                "artifact_root": str(tmp_path / "artifacts"),
                "gnn": {
                    "model_name": "stub",
                    "dataset_path": str(GRAPHS_DATA),
                    "train_file": str(GRAPHS_DATA / "train.jsonl"),
                    "validation_file": str(GRAPHS_DATA / "validation.jsonl"),
                    "output_dir": str(tmp_path / "gnn"),
                },
            }
        )
    )
    captured = {}

    class FakeTask:
        def __init__(self, cfg, *_args, **_kwargs):
            captured["config"] = cfg

        def run(self):
            pass

    monkeypatch.setattr("sphana_trainer.cli.GNNTask", FakeTask)
    result = runner.invoke(
        app,
        ["train", "gnn", "--config", str(cfg_path), "--profile-steps", "3"],
    )
    assert result.exit_code == 0
    assert captured["config"].profile_steps == 3


def test_dataset_build_cli(monkeypatch, tmp_path):
    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir()
    (ingest_dir / "chunks.jsonl").write_text('{"id": "doc-chunk-0", "document_id": "doc", "text": "Sentence one. Sentence two."}\n')
    (ingest_dir / "relations.jsonl").write_text(
        json.dumps(
            {"doc_id": "doc", "chunk_id": "doc-chunk-0", "subject": "A", "object": "B", "predicate": "rel", "sentence": "A B", "confidence": 1.0}
        )
        + "\n"
    )
    parses_dir = ingest_dir / "cache" / "parses"
    parses_dir.mkdir(parents=True)
    (parses_dir / "doc-chunk-0.json").write_text('{"sentences": []}')
    extra = tmp_path / "extra.jsonl"
    extra.write_text("{}\n")

    captured = {}

    def fake_build(chunks, relations, output, **kwargs):
        captured["chunks"] = chunks
        captured["parses_dir"] = kwargs.get("parses_dir")
        class Result(SimpleNamespace):
            embedding_train = relation_train = gnn_train = 1
            embedding_val = relation_val = gnn_val = 0
            output_dir = output
        return Result()

    monkeypatch.setattr("sphana_trainer.cli.build_datasets_from_ingestion", fake_build)
    result = runner.invoke(
        app,
        [
            "dataset-build-from-ingest",
            str(ingest_dir),
            "--output-dir",
            str(tmp_path / "derived"),
            "--extra-embedding",
            str(extra),
            "--extra-relation",
            str(extra),
            "--extra-gnn",
            str(extra),
        ],
    )
    assert result.exit_code == 0
    assert captured["chunks"] == ingest_dir / "chunks.jsonl"
    assert captured["parses_dir"] == parses_dir


def test_dataset_build_cli_missing_files(tmp_path):
    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir()
    result = runner.invoke(app, ["dataset-build-from-ingest", str(ingest_dir)])
    assert result.exit_code != 0
    assert "chunks.jsonl" in (result.stderr or result.stdout)


def test_ingest_cache_models_requires_option():
    result = runner.invoke(app, ["ingest-cache-models"])
    assert result.exit_code != 0


def test_train_sweep_embedding(monkeypatch, tmp_path):
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": str(tmp_path),
                "artifact_root": str(tmp_path / "artifacts"),
                "embedding": {
                    "model_name": "stub",
                    "dataset_path": str(EMBED_DATA),
                    "train_file": str(EMBED_DATA / "train.jsonl"),
                    "validation_file": str(EMBED_DATA / "validation.jsonl"),
                    "output_dir": str(tmp_path / "out"),
                },
            }
        )
    )

    runs = []

    class FakeTask:
        def __init__(self, cfg, *_args, **_kwargs):
            runs.append(cfg.learning_rate)

        def run(self):
            pass

    monkeypatch.setattr("sphana_trainer.cli.EmbeddingTask", FakeTask)
    result = runner.invoke(
        app,
        [
            "train",
            "sweep",
            "embedding",
            "--config",
            str(cfg_path),
            "--lr",
            "1e-5",
            "--lr",
            "2e-5",
            "--temperature",
            "0.05",
        ],
    )
    assert result.exit_code == 0
    assert runs == [1e-5, 2e-5]
    summary = tmp_path / "artifacts" / "sweeps" / "embedding.jsonl"
    assert summary.exists()


def test_train_sweep_invalid_component(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("workspace_dir: .\nartifact_root: .\nembedding: {}\n")
    result = runner.invoke(app, ["train", "sweep", "invalid", "--config", str(cfg_path)])
    assert result.exit_code != 0


def test_train_sweep_missing_section(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("workspace_dir: .\nartifact_root: .\n")
    result = runner.invoke(app, ["train", "sweep", "embedding", "--config", str(cfg_path)])
    assert result.exit_code != 0


def test_train_sweep_relation_and_gnn(monkeypatch, tmp_path):
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": str(tmp_path),
                "artifact_root": str(tmp_path / "artifacts"),
                "relation": {
                    "model_name": "stub",
                    "dataset_path": str(RELATION_DATA),
                    "train_file": str(RELATION_DATA / "train.jsonl"),
                    "validation_file": str(RELATION_DATA / "validation.jsonl"),
                    "output_dir": str(tmp_path / "rel"),
                },
                "gnn": {
                    "model_name": "stub",
                    "dataset_path": str(GRAPHS_DATA),
                    "train_file": str(GRAPHS_DATA / "train.jsonl"),
                    "validation_file": str(GRAPHS_DATA / "validation.jsonl"),
                    "output_dir": str(tmp_path / "gnn"),
                },
            }
        )
    )

    runs = []

    class FakeRelationTask:
        def __init__(self, cfg, *_args, **_kwargs):
            runs.append(("relation", cfg.learning_rate if hasattr(cfg, "learning_rate") else None))

        def run(self):
            pass

    class FakeGNNTask:
        def __init__(self, cfg, *_args, **_kwargs):
            runs.append(("gnn", cfg.hidden_dim if hasattr(cfg, "hidden_dim") else None))

        def run(self):
            pass

    monkeypatch.setattr("sphana_trainer.cli.RelationExtractionTask", FakeRelationTask)
    result = runner.invoke(
        app,
        ["train", "sweep", "relation", "--config", str(cfg_path), "--lr", "1e-5"],
    )
    assert result.exit_code == 0
    assert ("relation", 1e-05) in runs

    monkeypatch.setattr("sphana_trainer.cli.GNNTask", FakeGNNTask)
    result = runner.invoke(
        app,
        ["train", "sweep", "gnn", "--config", str(cfg_path), "--hidden-dim", "64"],
    )
    assert result.exit_code == 0
    assert ("gnn", 64) in runs


def test_load_sweep_grid_file_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n")
    with pytest.raises(typer.BadParameter):
        _load_sweep_grid_file(bad)

    grid_file = tmp_path / "grid.yaml"
    grid_file.write_text('learning_rate:\n  - "1e-5"\n')
    grid = _load_sweep_grid_file(grid_file)
    assert pytest.approx(grid["learning_rate"][0]) == 1e-05

    single = tmp_path / "single.yaml"
    single.write_text("batch_size: 8\n")
    grid = _load_sweep_grid_file(single)
    assert grid["batch_size"] == [8]

    weird = tmp_path / "weird.yaml"
    weird.write_text('learning_rate:\n  - "fast"\n')
    grid = _load_sweep_grid_file(weird)
    assert grid["learning_rate"][0] == "fast"


def test_train_sweep_with_grid_file(monkeypatch, tmp_path):
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": str(tmp_path),
                "artifact_root": str(tmp_path / "artifacts"),
                "embedding": {
                    "model_name": "stub",
                    "dataset_path": str(EMBED_DATA),
                    "train_file": str(EMBED_DATA / "train.jsonl"),
                    "validation_file": str(EMBED_DATA / "validation.jsonl"),
                    "output_dir": str(tmp_path / "out"),
                },
            }
        )
    )
    grid_file = tmp_path / "grid.yaml"
    grid_file.write_text("learning_rate:\n  - 1e-5\nbatch_size:\n  - 4\n")
    runs = []

    class FakeTask:
        def __init__(self, cfg, *_args, **_kwargs):
            runs.append((cfg.learning_rate, cfg.batch_size))

        def run(self):
            return SimpleNamespace(checkpoint_dir=tmp_path, onnx_path=tmp_path / "model.onnx", metrics={"val_cosine": 0.9})

    monkeypatch.setattr("sphana_trainer.cli.EmbeddingTask", FakeTask)
    result = runner.invoke(
        app,
        [
            "train",
            "sweep",
            "embedding",
            "--config",
            str(cfg_path),
            "--grid-file",
            str(grid_file),
        ],
    )
    assert result.exit_code == 0
    assert runs == [(1e-05, 4)]


def test_train_sweep_embedding_handles_unknown_override(monkeypatch, tmp_path):
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": str(tmp_path),
                "artifact_root": str(tmp_path / "artifacts"),
                "embedding": {
                    "model_name": "stub",
                    "dataset_path": str(EMBED_DATA),
                    "train_file": str(EMBED_DATA / "train.jsonl"),
                    "validation_file": str(EMBED_DATA / "validation.jsonl"),
                    "output_dir": str(tmp_path / "out"),
                },
            }
        )
    )

    class FakeTask:
        def __init__(self, cfg, *_args, **_kwargs):
            self.cfg = cfg

        def run(self):
            pass

    monkeypatch.setattr("sphana_trainer.cli.EmbeddingTask", FakeTask)
    result = runner.invoke(
        app,
        ["train", "sweep", "embedding", "--config", str(cfg_path), "--hidden-dim", "64"],
    )
    assert result.exit_code == 0


def test_train_sweep_relation_defaults(monkeypatch, tmp_path):
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": str(tmp_path),
                "artifact_root": str(tmp_path / "artifacts"),
                "relation": {
                    "model_name": "stub",
                    "dataset_path": str(RELATION_DATA),
                    "train_file": str(RELATION_DATA / "train.jsonl"),
                    "validation_file": str(RELATION_DATA / "validation.jsonl"),
                    "output_dir": str(tmp_path / "rel"),
                },
            }
        )
    )
    invoked = {}

    class FakeRelationTask:
        def __init__(self, cfg, *_args, **_kwargs):
            invoked["run"] = cfg

        def run(self):
            pass

    monkeypatch.setattr("sphana_trainer.cli.RelationExtractionTask", FakeRelationTask)
    result = runner.invoke(app, ["train", "sweep", "relation", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert "run" in invoked


def test_build_sweep_grid_handles_empty():
    assert _build_sweep_grid({}) == [{}]


def test_should_run_stage_behaviour(tmp_path):
    state = {"stages": {}}
    assert _should_run_stage(state, "stage", False, None, False, set()) is False
    assert _should_run_stage(state, "stage", True, None, True, set()) is True
    output = tmp_path / "out"
    output.write_text("done")
    state["stages"]["stage"] = {"output": str(output), "status": "succeeded"}
    assert _should_run_stage(state, "stage", True, output, False, set()) is False


def test_cache_relation_model_success(monkeypatch):
    class FakeTokenizer:
        @staticmethod
        def from_pretrained(name):
            FakeTokenizer.called = name

    class FakeModel:
        @staticmethod
        def from_pretrained(name):
            FakeModel.called = name

    module = SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModelForSequenceClassification=FakeModel)
    monkeypatch.setitem(sys.modules, "transformers", module)
    _cache_relation_model("model")
    assert FakeTokenizer.called == "model"
    assert FakeModel.called == "model"


def test_download_spacy_model_success(monkeypatch):
    class FakeCLI:
        called = None

        @staticmethod
        def download(name):
            FakeCLI.called = name

    module = SimpleNamespace(cli=SimpleNamespace(download=FakeCLI.download))
    monkeypatch.setitem(sys.modules, "spacy", module)
    monkeypatch.setitem(sys.modules, "spacy.cli", module.cli)
    _download_spacy_model("en_core_web_sm")
    assert FakeCLI.called == "en_core_web_sm"


def test_download_stanza_success(monkeypatch):
    class FakePipeline:
        downloads = []

        @staticmethod
        def download(lang):
            FakePipeline.downloads.append(lang)

    class FakeStanza:
        @staticmethod
        def download(lang):
            FakePipeline.downloads.append(lang)

        @staticmethod
        def Pipeline(**kwargs):
            return kwargs

    monkeypatch.setitem(sys.modules, "stanza", FakeStanza)
    _download_stanza("en")
    assert "en" in FakePipeline.downloads or FakePipeline.downloads == ["en"]


def test_cache_relation_model_missing_dependency(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(typer.BadParameter):
        _cache_relation_model("model")


def test_download_spacy_model_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy", None)
    monkeypatch.setitem(sys.modules, "spacy.cli", None)
    with pytest.raises(typer.BadParameter):
        _download_spacy_model("en_core_web_sm")


def test_download_stanza_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "stanza", None)
    with pytest.raises(typer.BadParameter):
        _download_stanza("en")


def test_export_and_package_commands(monkeypatch, tmp_path):
    cfg = tmp_path / "export.yaml"
    cfg.write_text("export:\n")
    called = {"export": False, "package": False}

    class DummyCfg(SimpleNamespace):
        manifest_path = str(tmp_path / "manifest.json")
        include_models = ["embedding"]
        publish_uri = None

    def fake_resolve(path, section):
        return DummyCfg(), tmp_path

    class FakeExportTask:
        def __init__(self, cfg, root):
            assert root == tmp_path

        def run(self):
            called["export"] = True

    class FakePackageTask(FakeExportTask):
        def run(self):
            called["package"] = True

    monkeypatch.setattr("sphana_trainer.cli._resolve_component_config", fake_resolve)
    monkeypatch.setattr("sphana_trainer.cli.ExportTask", FakeExportTask)
    result = runner.invoke(app, ["export", "--config", str(cfg)])
    assert result.exit_code == 0
    assert called["export"]

    monkeypatch.setattr("sphana_trainer.cli.PackageTask", FakePackageTask)
    result = runner.invoke(app, ["package", "--config", str(cfg)])
    assert result.exit_code == 0
    assert called["package"]

