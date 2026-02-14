"""Exception hierarchy for the partitioned distributed cache."""

from __future__ import annotations


class DistributedCacheError(Exception):
    """Base exception for all distributed cache errors."""

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(message)


# ── Cluster Errors ────────────────────────────────────────────────

class ClusterNotReadyError(DistributedCacheError):
    """Raised when an operation is attempted before the cluster is ready."""

    def __init__(self, message: str = "Cluster is not ready. No coordinator or partition table available.") -> None:
        super().__init__(message)


class CoordinatorUnavailableError(DistributedCacheError):
    """Raised when the coordinator cannot be reached."""

    def __init__(self, message: str = "Coordinator is unavailable.") -> None:
        super().__init__(message)


class NodeNotFoundError(DistributedCacheError):
    """Raised when a target node cannot be found in the cluster."""

    def __init__(self, address: str) -> None:
        self.address = address
        super().__init__(f"Node not found: {address}")


# ── Partition Errors ──────────────────────────────────────────────

class PartitionNotOwnedError(DistributedCacheError):
    """Raised when this node does not own the requested partition."""

    def __init__(self, partition_id: int, owner_address: str | None = None) -> None:
        self.partition_id = partition_id
        self.owner_address = owner_address
        msg = f"Partition {partition_id} is not owned by this node."
        if owner_address:
            msg += f" Owner: {owner_address}"
        super().__init__(msg)


class PartitionMigrationError(DistributedCacheError):
    """Raised when partition migration fails."""

    def __init__(self, partition_id: int, reason: str = "") -> None:
        self.partition_id = partition_id
        msg = f"Failed to migrate partition {partition_id}."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)


# ── Cache Errors ──────────────────────────────────────────────────

class CacheKeyNotFoundError(DistributedCacheError):
    """Raised when a cache key is not found."""

    def __init__(self, key: str, collection: str = "__default__") -> None:
        self.key = key
        self.collection = collection
        super().__init__(f"Key '{key}' not found in collection '{collection}'.")


class CollectionNotFoundError(DistributedCacheError):
    """Raised when a collection does not exist."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Collection '{name}' not found.")


class CacheCapacityError(DistributedCacheError):
    """Raised when a collection has reached its max capacity and eviction fails."""

    def __init__(self, collection: str, max_entries: int) -> None:
        self.collection = collection
        self.max_entries = max_entries
        super().__init__(
            f"Collection '{collection}' has reached max capacity of {max_entries} entries."
        )


# ── Lock Errors ───────────────────────────────────────────────────

class LockAcquireTimeoutError(DistributedCacheError):
    """Raised when a lock cannot be acquired within the timeout."""

    def __init__(
        self,
        key: str,
        timeout: float,
        current_owner: str | None = None,
    ) -> None:
        self.key = key
        self.timeout = timeout
        self.current_owner = current_owner
        msg = f"Failed to acquire lock '{key}' within {timeout:.1f}s."
        if current_owner:
            msg += f" Current owner: {current_owner}"
        super().__init__(msg)


class LockHoldTimeoutError(DistributedCacheError):
    """Raised when a lock has expired while being held."""

    def __init__(self, key: str, hold_timeout: float) -> None:
        self.key = key
        self.hold_timeout = hold_timeout
        super().__init__(
            f"Lock '{key}' expired after hold timeout of {hold_timeout:.1f}s."
        )


class LockNotHeldError(DistributedCacheError):
    """Raised when attempting to release a lock not held by this owner."""

    def __init__(self, key: str, owner_id: str) -> None:
        self.key = key
        self.owner_id = owner_id
        super().__init__(
            f"Lock '{key}' is not held by owner '{owner_id}'."
        )


# ── Network / RPC Errors ─────────────────────────────────────────

class RpcError(DistributedCacheError):
    """Raised when an RPC call fails."""

    def __init__(self, target_address: str, reason: str = "") -> None:
        self.target_address = target_address
        msg = f"RPC to {target_address} failed."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)


class RpcTimeoutError(RpcError):
    """Raised when an RPC call times out."""

    def __init__(self, target_address: str, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(target_address, reason=f"Timed out after {timeout:.1f}s")


class RpcConnectionError(RpcError):
    """Raised when a TCP connection to a peer cannot be established."""

    def __init__(self, target_address: str, reason: str = "") -> None:
        super().__init__(target_address, reason=reason or "Connection refused or unreachable")


# ── Storage / PVC Errors ─────────────────────────────────────────

class PvcStorageError(DistributedCacheError):
    """Raised when a PVC read/write operation fails."""

    def __init__(self, path: str, reason: str = "") -> None:
        self.path = path
        msg = f"PVC storage error at '{path}'."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)

# ── Quorum / Split-Brain Errors ───────────────────────────────────

class QuorumLostError(DistributedCacheError):
    """Raised when the node cannot see a majority of cluster members."""

    def __init__(
        self,
        message: str | None = None,
        visible_nodes: int = 0,
        total_nodes: int = 0,
    ) -> None:
        self.visible_nodes = visible_nodes
        self.total_nodes = total_nodes
        if message is None:
            message = (
                f"Quorum lost: only {visible_nodes}/{total_nodes} nodes visible. "
                f"Writes are rejected until quorum is restored."
            )
        super().__init__(message)

class WalCorruptionError(DistributedCacheError):
    """Raised when a WAL record fails checksum validation."""

    def __init__(self, segment: str, offset: int, reason: str = "") -> None:
        self.segment = segment
        self.offset = offset
        msg = f"WAL corruption in segment '{segment}' at offset {offset}."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)
