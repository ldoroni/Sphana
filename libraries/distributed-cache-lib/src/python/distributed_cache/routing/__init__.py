"""Task routing — partition-aware distributed task dispatch."""

from .task_result import TaskResult, TaskStatus
from .task_router import DistributedTaskRouter

__all__ = [
    "DistributedTaskRouter",
    "TaskResult",
    "TaskStatus",
]