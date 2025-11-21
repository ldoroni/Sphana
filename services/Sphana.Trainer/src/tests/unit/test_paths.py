from pathlib import Path

from sphana_trainer.utils.paths import prepare_run_directory, prune_run_directories, resolve_split_file


def test_resolve_split_file(tmp_path):
    base = tmp_path / "dataset"
    base.mkdir()
    (base / "train.jsonl").write_text("[]")
    assert resolve_split_file(base, None, "train") == base / "train.jsonl"
    assert resolve_split_file(base, None, "validation") is None
    override = tmp_path / "custom.jsonl"
    override.write_text("[]")
    assert resolve_split_file(base, override, "train") == override


def test_prepare_and_prune_runs(tmp_path):
    comp_dir = tmp_path / "embedding"
    run1 = prepare_run_directory(comp_dir, version="v1")
    run2 = prepare_run_directory(comp_dir, version="v2")
    assert run1.exists() and run2.exists()
    prune_run_directories(comp_dir, keep=1)
    remaining = [p.name for p in comp_dir.iterdir()]
    assert len(remaining) == 1


def test_prune_run_directories_keep_zero(tmp_path):
    comp_dir = tmp_path / "embedding"
    comp_dir.mkdir()
    prune_run_directories(comp_dir, keep=0)

