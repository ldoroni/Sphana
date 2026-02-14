"""Task result models for distributed task routing."""

from __future__ import annotations

import enum
import time
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, enum.Enum):
    """Status of a routed task execution."""

    COMPLETED = "completed"
    FAILED = "failed"


class TaskResult(BaseModel):
    """Result returned from a routed task execution.

    Attributes:
        status: Whether the task completed or failed.
        response: The response payload from the handler (if completed).
        error: Error message (if failed).
        node_address: The address of the node that executed the task.
        timestamp: When the result was produced.
    """

    status: TaskStatus = TaskStatus.COMPLETED
    """Whether the task completed or failed."""

    response: dict[str, Any] = Field(default_factory=dict)
    """The response payload from the handler."""

    error: str | None = None
    """Error message if the task failed."""

    node_address: str = ""
    """Address of the node that executed the task."""

    timestamp: float = Field(default_factory=time.time)
    """Epoch timestamp when the result was produced."""

    @property
    def is_success(self) -> bool:
        """Check if the task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    @staticmethod
    def success(response: dict[str, Any] | None = None, node_address: str = "") -> TaskResult:
        """Create a successful task result."""
        return TaskResult(
            status=TaskStatus.COMPLETED,
            response=response or {},
            node_address=node_address,
        )

    @staticmethod
    def failure(error: str, node_address: str = "") -> TaskResult:
        """Create a failed task result."""
        return TaskResult(
            status=TaskStatus.FAILED,
            error=error,
            node_address=node_address,
        )