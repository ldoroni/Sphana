"""Tests for task helpers."""

import json
import shutil
from pathlib import Path

import pytest

from sphana_trainer.tasks.base import BaseTask
from sphana_trainer.tasks.package import PackageTask

TARGET_DIR = Path("target/test-task-runs")


@pytest.fixture
def task_output_dir():
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR, ignore_errors=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    yield TARGET_DIR
    shutil.rmtree(TARGET_DIR, ignore_errors=True)


class DummyTask(BaseTask):
    def run(self):
        self._simulate_stage("TEST", "running")
        return self._prepare_output_dir()


def test_base_task_helpers(task_output_dir):
    cfg = type("Cfg", (), {"output_dir": task_output_dir})
    task = DummyTask(config=cfg())
    output = task._prepare_output_dir()
    assert output.exists()
    task._simulate_stage("STEP", "message")


def test_package_task_missing_manifest(task_output_dir):
    cfg = type("Cfg", (), {"manifest_path": task_output_dir / "missing.json"})
    task = PackageTask(cfg, artifact_root=task_output_dir)
    with pytest.raises(FileNotFoundError):
        task.run()


def test_package_task_missing_artifact(task_output_dir):
    manifest_path = task_output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"components": ["embedding"], "artifacts": {"embedding": str(task_output_dir / "file.onnx")}})
    )
    cfg = type("Cfg", (), {"manifest_path": manifest_path})
    task = PackageTask(cfg, artifact_root=task_output_dir)
    with pytest.raises(FileNotFoundError):
        task.run()

