"""Integration-style test for dataset build CLI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from sphana_trainer.cli import app


runner = CliRunner()


def test_dataset_build_cli_creates_files(tmp_path):
    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir()
    chunks = ingest_dir / "chunks.jsonl"
    relations = ingest_dir / "relations.jsonl"
    parse_dir = ingest_dir / "cache" / "parses"
    parse_dir.mkdir(parents=True)

    chunks.write_text('{"id": "doc-chunk-0", "document_id": "doc", "text": "Sentence one. Sentence two."}\n')
    relations.write_text(
        json.dumps(
            {
                "doc_id": "doc",
                "chunk_id": "doc-chunk-0",
                "subject": "Alice",
                "object": "NRDB",
                "predicate": "built",
                "sentence": "Alice built NRDB.",
                "confidence": 0.9,
            }
        )
        + "\n"
    )
    (parse_dir / "doc-chunk-0.json").write_text(json.dumps({"sentences": []}))

    output_dir = tmp_path / "derived"
    result = runner.invoke(
        app,
        [
            "dataset-build-from-ingest",
            str(ingest_dir),
            "--output-dir",
            str(output_dir),
            "--parses-dir",
            str(parse_dir),
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "embedding" / "train.jsonl").exists()
    assert (output_dir / "relation" / "train.jsonl").exists()
    assert (output_dir / "gnn" / "train.jsonl").exists()

