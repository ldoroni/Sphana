from datetime import datetime, timezone
from pathlib import Path

import pytest

from sphana_trainer.workflow.state import (
    WorkflowLock,
    clear_stage,
    load_workflow_state,
    record_stage_failure,
    record_stage_start,
    record_stage_success,
    stage_is_current,
)


def test_stage_is_current_without_output(tmp_path):
    state_path = tmp_path / "state.json"
    state = record_stage_success(state_path, "stage", None)
    assert stage_is_current(state, "stage", None) is True


def test_stage_is_current_with_missing_output(tmp_path):
    state = {"stages": {"stage": {"output": str(tmp_path / "missing"), "status": "succeeded"}}}
    assert stage_is_current(state, "stage", Path(tmp_path / "missing")) is False


def test_record_stage_status_flow(tmp_path):
    state_path = tmp_path / "state.json"
    record_stage_start(state_path, "stage")
    state = load_workflow_state(state_path)
    assert state["stages"]["stage"]["status"] == "running"
    record_stage_failure(state_path, "stage", "boom")
    state = load_workflow_state(state_path)
    assert state["stages"]["stage"]["status"] == "failed"
    record_stage_success(state_path, "stage", tmp_path)
    state = load_workflow_state(state_path)
    assert state["stages"]["stage"]["status"] == "succeeded"


def test_workflow_lock(tmp_path):
    lock_path = tmp_path / "lock.json"
    lock = WorkflowLock(lock_path)
    lock.acquire()
    with pytest.raises(RuntimeError):
        WorkflowLock(lock_path).acquire()
    lock.release()
    assert not lock_path.exists()


def test_workflow_lock_invalid_metadata(tmp_path):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{bad json")
    lock = WorkflowLock(lock_path)
    lock.acquire()
    lock.release()


def test_workflow_lock_context_manager(tmp_path):
    lock_path = tmp_path / "lock.json"
    with WorkflowLock(lock_path) as lock:
        assert lock.path.exists()
    assert not lock_path.exists()


def test_workflow_lock_detects_recent_timestamp(tmp_path):
    lock_path = tmp_path / "lock.json"
    now = datetime.now(timezone.utc)
    lock_path.write_text(
        '{"timestamp": "%s"}' % (now.isoformat(),)
    )
    lock = WorkflowLock(lock_path, stale_minutes=30)
    with pytest.raises(RuntimeError):
        lock.acquire()


def test_clear_stage_removes_entry(tmp_path):
    state_path = tmp_path / "state.json"
    record_stage_success(state_path, "stage", None)
    state = clear_stage(state_path, "stage")
    assert "stage" not in state["stages"]


def test_load_workflow_state_missing(tmp_path):
    state = load_workflow_state(tmp_path / "missing.json")
    assert state == {"stages": {}}

