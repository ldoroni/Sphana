from pathlib import Path

from sphana_trainer.utils.fingerprint import dataset_fingerprint


def test_dataset_fingerprint_file(tmp_path):
    file_path = tmp_path / "data.jsonl"
    file_path.write_text("hello")
    fp = dataset_fingerprint(file_path)
    assert isinstance(fp, str) and len(fp) == 64


def test_dataset_fingerprint_directory(tmp_path):
    dir_path = tmp_path / "dir"
    nested = dir_path / "nested"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_text("content")
    fp_dir = dataset_fingerprint(dir_path)
    assert len(fp_dir) == 64


def test_dataset_fingerprint_missing(tmp_path):
    missing = tmp_path / "missing"
    fp = dataset_fingerprint(missing)
    assert len(fp) == 64


def test_dataset_fingerprint_handles_unreadable(monkeypatch, tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("content")

    original_open = Path.open

    def fake_open(self, *args, **kwargs):
        if self == file_path:
            raise OSError("boom")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)
    fp = dataset_fingerprint(file_path)
    assert len(fp) == 64

