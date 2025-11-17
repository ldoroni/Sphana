"""Artifact publishing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests


def publish_manifest(manifest_path: Path, endpoint: str, timeout: int = 10) -> str:
    """POST the manifest JSON to an external service."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

    response = requests.post(
        endpoint,
        json={"manifest": manifest_path.read_text()},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text

