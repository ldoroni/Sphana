"""Distributed Cache Library v2 — Partitioned, Hazelcast-style.

A production-ready, embedded distributed cache for Python applications
running across multiple Kubernetes pods (or any networked nodes).
Uses hash-based partitioning with automatic rebalancing, coordinator
election (oldest-member strategy), TCP-based RPC, and distributed
locking with fencing tokens.

Quick Start::

    from distributed_cache import DistributedCache, ClusterNodesProvider

    class K8sProvider(ClusterNodesProvider):
        def get_node_addresses(self) -> list[str]:
            # Return current pod addresses from Kubernetes API
            return ["10.0.0.1:9100", "10.0.0.2:9100"]

    cache = DistributedCache(
        self_address="10.0.0.1:9100",
        cluster_nodes_provider=K8sProvider(),
    )
    cache.start()

    # Cache operations
    cache.put("key", "value", ttl=60)
    val = cache.get("key")

    # Named collections
    sessions = cache.get_collection("sessions")
    sessions.put("user:1", {"name": "Alice"})

    # Distributed locking
    with cache.acquire_lock("resource-x"):
        pass  # critical section

    cache.stop()
"""

from .cache_collection import CacheCollection
from .cluster.cluster_nodes_provider import ClusterNodesProvider
from .distributed_cache import DistributedCache
from .exceptions import (
    CacheCapacityError,
    CacheKeyNotFoundError,
    ClusterNotReadyError,
    CollectionNotFoundError,
    CoordinatorUnavailableError,
    DistributedCacheError,
    LockAcquireTimeoutError,
    LockHoldTimeoutError,
    LockNotHeldError,
    PartitionMigrationError,
    PartitionNotOwnedError,
    QuorumLostError,
    RpcConnectionError,
    RpcError,
    RpcTimeoutError,
    WalCorruptionError,
)
from .locking.lock_handle import LockHandle
from .routing.task_result import TaskResult, TaskStatus
from .routing.task_router import DistributedTaskRouter
from .models import (
    CacheEntry,
    LockEntry,
    NodeInfo,
    NodeState,
    RpcMessageType,
    RpcResponse,
)

__all__ = [
    # Main entry point
    "DistributedCache",
    # Cluster
    "ClusterNodesProvider",
    # Collections
    "CacheCollection",
    # Locking
    "LockHandle",
    # Task routing
    "DistributedTaskRouter",
    "TaskResult",
    "TaskStatus",
    # Models
    "CacheEntry",
    "LockEntry",
    "NodeInfo",
    "NodeState",
    "RpcMessageType",
    "RpcResponse",
    # Exceptions
    "CacheCapacityError",
    "CacheKeyNotFoundError",
    "ClusterNotReadyError",
    "CollectionNotFoundError",
    "CoordinatorUnavailableError",
    "DistributedCacheError",
    "LockAcquireTimeoutError",
    "LockHoldTimeoutError",
    "LockNotHeldError",
    "PartitionMigrationError",
    "PartitionNotOwnedError",
    "QuorumLostError",
    "RpcConnectionError",
    "RpcError",
    "RpcTimeoutError",
    "WalCorruptionError",
]
