import json
from pathlib import Path

import pytest

from sphana_trainer.data.validation import dataset_statistics, validate_dataset_file, _iter_jsonl


@pytest.fixture
def schema_path(tmp_path):
    path = tmp_path / "schema.json"
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["text"],
        "properties": {"text": {"type": "string"}},
    }
    path.write_text(json.dumps(schema))
    return path


def test_validate_dataset_file_core(tmp_path, schema_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text": "ok"}\n')
    assert validate_dataset_file(data_path, schema_path) == 1


def test_validate_dataset_file_limit(tmp_path, schema_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text": "ok"}\n{"text": "skip"}\n')
    assert validate_dataset_file(data_path, schema_path, limit=1) == 2


def test_validate_dataset_file_error(tmp_path, schema_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"missing": "field"}\n')
    with pytest.raises(ValueError):
        validate_dataset_file(data_path, schema_path)


def test_dataset_statistics_core(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text": "foo bar", "label": "x"}\n')
    stats = dataset_statistics(data_path)
    assert stats["records"] == 1
    assert stats["labels"]["x"] == 1


def test_dataset_statistics_limit(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text": "foo"}\n{"text": "bar"}\n')
    stats = dataset_statistics(data_path, limit=1)
    assert stats["records"] == 1


def test_validate_dataset_file_missing(tmp_path, schema_path):
    with pytest.raises(FileNotFoundError):
        validate_dataset_file(tmp_path / "missing.jsonl", schema_path)


def test_validate_dataset_schema_missing(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text": "ok"}\n')
    with pytest.raises(FileNotFoundError):
        validate_dataset_file(data_path, tmp_path / "missing.json")


def test_dataset_statistics_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset_statistics(tmp_path / "missing.jsonl")


def test_dataset_statistics_empty(tmp_path):
    data_path = tmp_path / "empty.jsonl"
    data_path.write_text("")
    stats = dataset_statistics(data_path)
    assert stats["length"]["min"] == 0.0


def test_iter_jsonl_skips_blanks(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "foo"}\n\n')
    rows = list(_iter_jsonl(path))
    assert len(rows) == 1

