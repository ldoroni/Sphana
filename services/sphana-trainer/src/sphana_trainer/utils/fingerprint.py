"""Utilities for computing dataset fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path


def dataset_fingerprint(path: Path) -> str:
    """Create a stable fingerprint for a dataset file or directory."""

    path = path.expanduser().resolve()
    hasher = hashlib.sha256()

    if path.is_file():
        _update_hash_with_file(hasher, path)
    elif path.is_dir():
        for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
            _update_hash_with_file(hasher, file_path, include_name=True)
    else:
        hasher.update(str(path).encode("utf-8"))

    return hasher.hexdigest()


def _update_hash_with_file(hasher: "hashlib._Hash", file_path: Path, include_name: bool = False) -> None:
    if include_name:
        hasher.update(str(file_path.relative_to(file_path.anchor)).encode("utf-8"))
    try:
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        hasher.update(b"<unreadable>")
    stat = file_path.stat()
    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    hasher.update(str(stat.st_size).encode("utf-8"))

