"""Packaging task for release bundles."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from pathlib import Path

from loguru import logger

from sphana_trainer.tasks.base import BaseTask


class PackageTask(BaseTask):
    """Create a compressed archive of artifacts + manifest."""

    def __init__(self, config, artifact_root):
        super().__init__(config)
        self.artifact_root = Path(artifact_root)

    def run(self) -> None:
        manifest_path = Path(self.config.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        archive_path = manifest_path.with_suffix(".tar.gz")
        logger.info("Creating package {} with {} components", archive_path, len(manifest.get("components", [])))
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(manifest_path, arcname="manifest.json")
            for component, artifact_path in manifest.get("artifacts", {}).items():
                artifact_file = Path(artifact_path)
                if not artifact_file.exists():
                    raise FileNotFoundError(f"Artifact missing for {component}: {artifact_file}")
                tar.add(artifact_file, arcname=f"{component}/{artifact_file.name}")
        logger.success("Release bundle created at {}", archive_path)



