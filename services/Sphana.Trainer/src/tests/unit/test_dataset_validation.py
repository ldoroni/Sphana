"""Tests for dataset validation CLI."""

from pathlib import Path

from typer.testing import CliRunner

from sphana_trainer.cli import app


runner = CliRunner()


def test_dataset_validate_embedding(tmp_path):
    data = tmp_path / "embedding.jsonl"
    data.write_text('{"query": "q", "positive": "p"}\n')
    result = runner.invoke(app, ["dataset-validate", str(data), "--type", "embedding"])
    assert result.exit_code == 0
    assert "Validated" in result.stdout


def test_dataset_validate_fails_on_bad_record(tmp_path):
    data = tmp_path / "bad.jsonl"
    data.write_text('{"query": "only"}\n')
    result = runner.invoke(app, ["dataset-validate", str(data), "--type", "embedding"])
    assert result.exit_code != 0
    assert "validation" in result.stderr or "validation" in result.stdout


def test_dataset_stats_command(tmp_path):
    data = tmp_path / "stats.jsonl"
    data.write_text('{"query": "q", "positive": "p"}\n{"query": "another", "positive": "text"}\n')
    result = runner.invoke(app, ["dataset-stats", str(data)])
    assert result.exit_code == 0
    assert '"records": 2' in result.stdout


def test_dataset_validate_requires_type_or_schema(tmp_path):
    data = tmp_path / "embedding.jsonl"
    data.write_text('{"query": "q", "positive": "p"}\n')
    result = runner.invoke(app, ["dataset-validate", str(data)])
    assert result.exit_code != 0


def test_dataset_validate_invalid_type(tmp_path):
    data = tmp_path / "embedding.jsonl"
    data.write_text('{"query": "q", "positive": "p"}\n')
    result = runner.invoke(app, ["dataset-validate", str(data), "--type", "unknown"])
    assert result.exit_code != 0
    assert "Unsupported dataset type" in (result.stderr or result.stdout)


def test_dataset_validate_with_custom_schema(tmp_path):
    data = tmp_path / "embedding.jsonl"
    data.write_text('{"text": "ok"}\n')
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object", "required": ["text"]}')
    result = runner.invoke(app, ["dataset-validate", str(data), "--schema", str(schema)])
    assert result.exit_code == 0

