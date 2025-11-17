"""Export orchestration task."""

from __future__ import annotations

import json
from pathlib import Path

from pathlib import Path

from loguru import logger

from sphana_trainer.utils import load_artifact_metadata
from sphana_trainer.tasks.base import BaseTask


class ExportTask(BaseTask):
    """Aggregate artifacts into a manifest."""

    def __init__(self, config, artifact_root):
        super().__init__(config)
        self.artifact_root = Path(artifact_root)

    def run(self) -> None:
        manifest_path = Path(self.config.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "components": [],
            "publish_uri": self.config.publish_uri,
            "artifacts": {},
            "metadata": {},
        }
        for component in self.config.include_models:
            record = load_artifact_metadata(component, Path(self.artifact_root))
            artifact_path = record.quantized_path or record.onnx_path
            manifest["components"].append(component)
            manifest["artifacts"][component] = artifact_path
            manifest["metadata"][component] = {
                "version": record.version,
                "metrics": record.metrics,
                "onnx_path": record.onnx_path,
                "quantized_path": record.quantized_path,
            }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        logger.success("Manifest written to {}", manifest_path)



