"""Data models for the partitioned distributed cache.

All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

import enum
import time
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────


class NodeState(str, enum.Enum):
    """Lifecycle state of a cluster node."""

    JOINING = "joining"
    ACTIVE = "active"
    LEAVING = "leaving"
    DEAD = "dead"


class RpcMessageType(str, enum.Enum):
    """Types of RPC messages exchanged between nodes."""

    # Cache operations
    CACHE_PUT = "cache_put"
    CACHE_GET = "cache_get"
    CACHE_DELETE = "cache_delete"
    CACHE_EXISTS = "cache_exists"
    CACHE_KEYS = "cache_keys"
    CACHE_SIZE = "cache_size"
    CACHE_CLEAR = "cache_clear"
    CACHE_PUT_MANY = "cache_put_many"
    CACHE_GET_MANY = "cache_get_many"
    CACHE_DELETE_MANY = "cache_delete_many"

    # Collection operations
    COLLECTION_CREATE = "collection_create"
    COLLECTION_DELETE = "collection_delete"
    COLLECTION_EXISTS = "collection_exists"
    COLLECTION_LIST = "collection_list"

    # Lock operations
    LOCK_ACQUIRE = "lock_acquire"
    LOCK_RELEASE = "lock_release"
    LOCK_IS_LOCKED = "lock_is_locked"
    LOCK_FORCE_RELEASE = "lock_force_release"

    # Backup replication operations
    BACKUP_PUT = "backup_put"
    BACKUP_DELETE = "backup_delete"
    BACKUP_CLEAR = "backup_clear"
    BACKUP_SYNC = "backup_sync"

    # Cluster operations
    HEARTBEAT = "heartbeat"
    PARTITION_TABLE_UPDATE = "partition_table_update"
    PARTITION_LOAD = "partition_load"
    PARTITION_MIGRATE = "partition_migrate"
    PARTITION_MIGRATE_RECEIVE = "partition_migrate_receive"
    NODE_JOIN = "node_join"
    NODE_LEAVE = "node_leave"

    # Responses
    RESPONSE = "response"
    ERROR_RESPONSE = "error_response"


# ── Node & Cluster Models ─────────────────────────────────────────


class NodeInfo(BaseModel):
    """Information about a cluster node."""

    address: str
    """Node address in ``host:port`` format."""

    node_id: str
    """Unique identifier for this node instance."""

    state: NodeState = NodeState.JOINING
    """Current lifecycle state."""

    join_timestamp: float = Field(default_factory=time.time)
    """Epoch timestamp when this node joined the cluster."""

    last_heartbeat: float = Field(default_factory=time.time)
    """Epoch timestamp of the last received heartbeat."""

    rpc_port: int = 4322
    """TCP port for RPC communication."""


class PartitionOwnership(BaseModel):
    """Ownership info for a single partition."""

    partition_id: int
    """The partition number (0 to partition_count - 1)."""

    owner_address: str
    """Address of the primary owner node."""

    version: int = 0
    """Monotonically increasing version for conflict resolution."""


class PartitionTableModel(BaseModel):
    """Serializable partition table — maps partitions to owners."""

    version: int = 0
    """Monotonically increasing version for the entire table."""

    partition_count: int = 271
    """Total number of partitions."""

    assignments: dict[int, str] = Field(default_factory=dict)
    """Mapping of partition_id → owner_address."""

    coordinator_address: str | None = None
    """Address of the current coordinator node."""

    updated_at: float = Field(default_factory=time.time)
    """Epoch timestamp of last update."""


# ── Cache Models ──────────────────────────────────────────────────


class CacheEntry(BaseModel):
    """A single cached value with metadata."""

    key: str
    """The cache key."""

    value: Any
    """The cached value (must be msgpack-serializable)."""

    collection: str = "__default__"
    """The collection this entry belongs to."""

    created_at: float = Field(default_factory=time.time)
    """Epoch timestamp when the entry was created."""

    updated_at: float = Field(default_factory=time.time)
    """Epoch timestamp of the last update."""

    expires_at: float | None = None
    """Epoch timestamp when the entry expires. None = no expiry."""

    last_accessed_at: float = Field(default_factory=time.time)
    """Epoch timestamp of the last read access (for LRU)."""

    version: int = 1
    """Entry version, incremented on each update."""

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at

    def touch(self) -> None:
        """Update the last_accessed_at timestamp."""
        self.last_accessed_at = time.time()


class CollectionSettings(BaseModel):
    """Configuration for a named cache collection."""

    name: str
    """Collection name."""

    default_ttl: float | None = None
    """Default TTL in seconds for new entries. None = no expiry."""

    max_entries: int | None = None
    """Maximum number of entries (LRU eviction). None = unlimited."""

    created_at: float = Field(default_factory=time.time)
    """When the collection was created."""


# ── Lock Models ───────────────────────────────────────────────────


class LockEntry(BaseModel):
    """State of a distributed lock."""

    key: str
    """The qualified lock key (e.g., ``collection:key`` or ``__locks__:key``)."""

    owner_id: str
    """Identifier of the lock holder (``node_id:uuid``)."""

    fencing_token: int = 0
    """Monotonically increasing token for safe coordination."""

    acquired_at: float = Field(default_factory=time.time)
    """Epoch timestamp when the lock was acquired."""

    hold_timeout: float = 60.0
    """Maximum seconds the lock may be held."""

    @property
    def expires_at(self) -> float:
        """Epoch timestamp when this lock auto-expires."""
        return self.acquired_at + self.hold_timeout

    @property
    def is_expired(self) -> bool:
        """Check if this lock has exceeded its hold timeout."""
        return time.time() >= self.expires_at


class LockResult(BaseModel):
    """Result of a lock acquisition attempt."""

    acquired: bool = False
    """Whether the lock was successfully acquired."""

    owner_id: str | None = None
    """The current owner (caller if acquired, existing holder if not)."""

    fencing_token: int = 0
    """Fencing token (only meaningful if acquired)."""

    key: str = ""
    """The lock key."""


# ── RPC Models ────────────────────────────────────────────────────


class RpcMessage(BaseModel):
    """A message sent between nodes via TCP."""

    message_type: RpcMessageType
    """The type of this message."""

    request_id: str
    """Unique ID for request/response correlation."""

    sender_address: str
    """Address of the sending node."""

    payload: dict[str, Any] = Field(default_factory=dict)
    """Message-type-specific payload data."""

    timestamp: float = Field(default_factory=time.time)
    """When the message was created."""


class RpcResponse(BaseModel):
    """A response to an RPC request."""

    request_id: str
    """Matches the request_id of the original RpcMessage."""

    success: bool = True
    """Whether the operation succeeded."""

    payload: dict[str, Any] = Field(default_factory=dict)
    """Response data."""

    error: str | None = None
    """Error message if success is False."""

    timestamp: float = Field(default_factory=time.time)
    """When the response was created."""


# ── Partition Metadata (stored on PVC) ────────────────────────────


class PartitionMeta(BaseModel):
    """Metadata written to ``_meta.json`` in each partition directory."""

    partition_id: int
    """The partition number."""

    version: int = 0
    """Monotonic version, incremented on every write."""

    last_modified: float = Field(default_factory=time.time)
    """Epoch timestamp of the last modification."""

    entry_count: int = 0
    """Number of cache entries in this partition."""

    lock_count: int = 0
    """Number of active locks in this partition."""

    primary_owner: str | None = None
    """Address of the current primary owner."""


class CoordinatorInfo(BaseModel):
    """Coordinator identity written to ``coordinator.json`` on PVC."""

    address: str
    """Address of the coordinator node."""

    node_id: str
    """Node ID of the coordinator."""

    elected_at: float = Field(default_factory=time.time)
    """When this node became coordinator."""

    heartbeat_at: float = Field(default_factory=time.time)
    """Last heartbeat timestamp on PVC."""

    partition_table_version: int = 0
    """Version of the partition table managed by this coordinator."""