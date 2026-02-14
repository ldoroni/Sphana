"""Cache collection — logical grouping of key-value pairs within the cache.

Each collection has its own namespace, optional TTL and max-entry limits.
Operations are routed to the correct partition owner via the partition
strategy and table.

When backup replication and/or WAL persistence are configured, write
operations (put / delete / clear) are durably recorded before the
response is returned.

Features:
- LRU eviction when ``max_entries`` is reached.
- Read-from-backup fallback when the primary owner is unreachable.
- Quorum enforcement on write operations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .exceptions import CacheKeyNotFoundError, QuorumLostError
from .models import CacheEntry, RpcMessageType, RpcResponse
from .network.rpc_client import RpcClient
from .partitioning.partition_strategy import PartitionStrategy
from .partitioning.partition_table import PartitionTable
from .storage.backup_replicator import BackupReplicator
from .storage.partition_store import PartitionStore
from .storage.wal_store import WalStore

logger = logging.getLogger(__name__)


class CacheCollection:
    """A named collection of cached key-value entries.

    Parameters:
        name: Collection name.
        self_address: This node's ``host:port``.
        partition_strategy: Hashes keys to partition IDs.
        partition_table: Maps partitions to owner addresses.
        partition_store: Local partition store.
        rpc_client: For forwarding to remote owners.
        default_ttl: Default TTL in seconds (0 = no expiry).
        max_entries: Max entries per partition (0 = unlimited).
        backup_replicator: Optional replicator for in-memory backups.
        wal_store: Optional WAL for disk persistence.
        quorum_checker: Optional callable returning True if quorum is held.
    """

    def __init__(
        self,
        name: str,
        self_address: str,
        partition_strategy: PartitionStrategy,
        partition_table: PartitionTable,
        partition_store: PartitionStore,
        rpc_client: RpcClient,
        default_ttl: float = 0.0,
        max_entries: int = 0,
        backup_replicator: BackupReplicator | None = None,
        wal_store: WalStore | None = None,
        quorum_checker: Callable[[], bool] | None = None,
    ) -> None:
        self._name = name
        self._self_address = self_address
        self._strategy = partition_strategy
        self._table = partition_table
        self._store = partition_store
        self._rpc = rpc_client
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._backup = backup_replicator
        self._wal = wal_store
        self._quorum_checker = quorum_checker

    @property
    def name(self) -> str:
        return self._name

    # ── Quorum guard ──────────────────────────────────────────────

    def _check_quorum(self) -> None:
        """Raise ``QuorumLostError`` if the cluster lacks quorum.

        Only enforced when a quorum_checker is provided.
        """
        if self._quorum_checker is not None and not self._quorum_checker():
            raise QuorumLostError(
                "Write rejected: cluster quorum lost"
            )

    # ── LRU enforcement ───────────────────────────────────────────

    def _enforce_lru(self, partition_id: int) -> None:
        """Evict LRU entries if the collection exceeds max_entries."""
        if self._max_entries > 0:
            self._store.evict_lru(
                partition_id, self._name, self._max_entries
            )

    # ── Read-from-backup helper ───────────────────────────────────

    def _try_get_from_backup(
        self, partition_id: int, key: str, default: Any
    ) -> Any:
        """Attempt to read from backup nodes when the primary is
        unreachable.

        Returns the value if found on a backup node, otherwise
        *default*.
        """
        backup_nodes = self._table.get_backups(partition_id)
        for bnode in backup_nodes:
            if bnode == self._self_address:
                # Check local backup store
                value = self._store.get(partition_id, self._name, key)
                if value is not None:
                    return value
                continue
            try:
                resp = self._rpc.cache_get(bnode, self._name, key)
                if resp.success and resp.payload.get("found", False):
                    return resp.payload.get("value", default)
            except Exception:
                logger.debug(
                    "Backup read from %s failed for key '%s'",
                    bnode,
                    key,
                )
        return default

    # ── CRUD ──────────────────────────────────────────────────────

    def put(
        self, key: str, value: Any, ttl: float | None = None
    ) -> None:
        """Store a value.

        Args:
            key: Cache key.
            value: Any msgpack-serialisable value.
            ttl: Time-to-live in seconds. Uses collection default if None.

        Raises:
            QuorumLostError: If cluster quorum is lost.
        """
        self._check_quorum()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        partition_id = self._strategy.get_partition_id(key)
        owner = self._table.get_owner(partition_id)

        if owner is None or owner == self._self_address:
            # WAL first, then in-memory, then backup
            if self._wal and self._wal.enabled:
                self._wal.log_put(
                    partition_id, self._name, key, value, effective_ttl
                )
            self._store.put(
                partition_id, self._name, key, value, effective_ttl
            )
            self._enforce_lru(partition_id)
            if self._backup and self._backup.enabled:
                self._backup.replicate_put(
                    partition_id, self._name, key, value, effective_ttl
                )
            return

        try:
            self._rpc.cache_put(owner, self._name, key, value, effective_ttl)
        except Exception:
            logger.exception("Failed to put '%s' on %s", key, owner)
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value.

        Falls back to backup nodes if the primary owner is
        unreachable.

        Args:
            key: Cache key.
            default: Returned when key is not found.

        Returns:
            The cached value, or *default*.
        """
        partition_id = self._strategy.get_partition_id(key)
        owner = self._table.get_owner(partition_id)

        if owner is None or owner == self._self_address:
            value = self._store.get(partition_id, self._name, key)
            return value if value is not None else default

        try:
            resp = self._rpc.cache_get(owner, self._name, key)
            if resp.success:
                if resp.payload.get("found", True):
                    return resp.payload.get("value", default)
                return default
            return default
        except Exception:
            logger.warning(
                "Primary owner %s unreachable for get('%s'), "
                "trying backup nodes",
                owner,
                key,
            )
            return self._try_get_from_backup(partition_id, key, default)

    def delete(self, key: str) -> bool:
        """Delete a key.

        Returns:
            True if the key existed and was deleted.

        Raises:
            QuorumLostError: If cluster quorum is lost.
        """
        self._check_quorum()
        partition_id = self._strategy.get_partition_id(key)
        owner = self._table.get_owner(partition_id)

        if owner is None or owner == self._self_address:
            if self._wal and self._wal.enabled:
                self._wal.log_delete(partition_id, self._name, key)
            deleted = self._store.delete(partition_id, self._name, key)
            if deleted and self._backup and self._backup.enabled:
                self._backup.replicate_delete(
                    partition_id, self._name, key
                )
            return deleted

        try:
            resp = self._rpc.cache_delete(owner, self._name, key)
            return resp.success and resp.payload.get("deleted", False)
        except Exception:
            logger.exception("Failed to delete '%s' on %s", key, owner)
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        partition_id = self._strategy.get_partition_id(key)
        owner = self._table.get_owner(partition_id)

        if owner is None or owner == self._self_address:
            return self._store.exists(partition_id, self._name, key)

        try:
            resp = self._rpc.cache_exists(owner, self._name, key)
            return resp.success and resp.payload.get("exists", False)
        except Exception:
            logger.exception("Failed to check '%s' on %s", key, owner)
            return False

    # ── Bulk operations ───────────────────────────────────────────

    def put_many(
        self, items: dict[str, Any], ttl: float | None = None
    ) -> None:
        """Store multiple key-value pairs.

        Raises:
            QuorumLostError: If cluster quorum is lost.
        """
        self._check_quorum()
        effective_ttl = ttl if ttl is not None else self._default_ttl

        # Group by partition → by owner
        groups: dict[str, dict[str, Any]] = {}  # owner → {key: value}
        local_items: dict[int, dict[str, Any]] = {}  # partition → {key: value}

        for key, value in items.items():
            pid = self._strategy.get_partition_id(key)
            owner = self._table.get_owner(pid)
            if owner is None or owner == self._self_address:
                local_items.setdefault(pid, {})[key] = value
            else:
                groups.setdefault(owner, {})[key] = value

        # Local puts (WAL + store + backup)
        for pid, kv in local_items.items():
            for k, v in kv.items():
                if self._wal and self._wal.enabled:
                    self._wal.log_put(pid, self._name, k, v, effective_ttl)
                self._store.put(pid, self._name, k, v, effective_ttl)
                if self._backup and self._backup.enabled:
                    self._backup.replicate_put(
                        pid, self._name, k, v, effective_ttl
                    )
            self._enforce_lru(pid)

        # Remote puts
        for owner, kv in groups.items():
            try:
                self._rpc.cache_put_many(owner, self._name, kv, effective_ttl)
            except Exception:
                logger.exception(
                    "Failed to put_many (%d keys) on %s", len(kv), owner
                )

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Retrieve multiple values.

        Falls back to backup nodes for keys whose primary owner is
        unreachable.
        """
        result: dict[str, Any] = {}

        # Group by owner
        remote_groups: dict[str, list[str]] = {}
        for key in keys:
            pid = self._strategy.get_partition_id(key)
            owner = self._table.get_owner(pid)
            if owner is None or owner == self._self_address:
                value = self._store.get(pid, self._name, key)
                if value is not None:
                    result[key] = value
            else:
                remote_groups.setdefault(owner, []).append(key)

        for owner, remote_keys in remote_groups.items():
            try:
                resp = self._rpc.cache_get_many(
                    owner, self._name, remote_keys
                )
                if resp.success:
                    result.update(resp.payload.get("entries", {}))
            except Exception:
                logger.warning(
                    "Primary owner %s unreachable for get_many, "
                    "trying backup nodes for %d keys",
                    owner,
                    len(remote_keys),
                )
                # Fallback to backup for each key
                for key in remote_keys:
                    pid = self._strategy.get_partition_id(key)
                    value = self._try_get_from_backup(pid, key, None)
                    if value is not None:
                        result[key] = value

        return result

    def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys. Returns count of deleted keys.

        Raises:
            QuorumLostError: If cluster quorum is lost.
        """
        self._check_quorum()
        deleted = 0

        remote_groups: dict[str, list[str]] = {}
        for key in keys:
            pid = self._strategy.get_partition_id(key)
            owner = self._table.get_owner(pid)
            if owner is None or owner == self._self_address:
                if self._wal and self._wal.enabled:
                    self._wal.log_delete(pid, self._name, key)
                if self._store.delete(pid, self._name, key):
                    deleted += 1
                    if self._backup and self._backup.enabled:
                        self._backup.replicate_delete(
                            pid, self._name, key
                        )
            else:
                remote_groups.setdefault(owner, []).append(key)

        for owner, remote_keys in remote_groups.items():
            try:
                resp = self._rpc.cache_delete_many(
                    owner, self._name, remote_keys
                )
                if resp.success:
                    deleted += resp.payload.get("deleted_count", 0)
            except Exception:
                logger.exception(
                    "Failed to delete_many (%d keys) on %s",
                    len(remote_keys),
                    owner,
                )

        return deleted

    # ── Collection-level ops ──────────────────────────────────────

    def keys(self) -> list[str]:
        """Return all keys in this collection (across all partitions)."""
        all_keys: list[str] = []

        # Local partitions
        local_partitions = self._table.get_partitions_for_node(
            self._self_address
        )
        for pid in local_partitions:
            all_keys.extend(self._store.keys(pid, self._name))

        # Remote partitions
        seen_owners: set[str] = set()
        for pid in range(self._strategy.partition_count):
            owner = self._table.get_owner(pid)
            if (
                owner is not None
                and owner != self._self_address
                and owner not in seen_owners
            ):
                seen_owners.add(owner)
                try:
                    resp = self._rpc.cache_keys(owner, self._name)
                    if resp.success:
                        all_keys.extend(resp.payload.get("keys", []))
                except Exception:
                    logger.debug("Failed to get keys from %s", owner)

        return all_keys

    def size(self) -> int:
        """Return total entry count in this collection."""
        total = 0

        local_partitions = self._table.get_partitions_for_node(
            self._self_address
        )
        for pid in local_partitions:
            total += self._store.size(pid, self._name)

        seen_owners: set[str] = set()
        for pid in range(self._strategy.partition_count):
            owner = self._table.get_owner(pid)
            if (
                owner is not None
                and owner != self._self_address
                and owner not in seen_owners
            ):
                seen_owners.add(owner)
                try:
                    resp = self._rpc.cache_size(owner, self._name)
                    if resp.success:
                        total += resp.payload.get("size", 0)
                except Exception:
                    logger.debug("Failed to get size from %s", owner)

        return total

    def clear(self) -> None:
        """Remove all entries from this collection.

        Raises:
            QuorumLostError: If cluster quorum is lost.
        """
        self._check_quorum()
        local_partitions = self._table.get_partitions_for_node(
            self._self_address
        )
        for pid in local_partitions:
            if self._wal and self._wal.enabled:
                self._wal.log_clear(pid, self._name)
            self._store.clear_collection(pid, self._name)
            if self._backup and self._backup.enabled:
                self._backup.replicate_clear(pid, self._name)

        seen_owners: set[str] = set()
        for pid in range(self._strategy.partition_count):
            owner = self._table.get_owner(pid)
            if (
                owner is not None
                and owner != self._self_address
                and owner not in seen_owners
            ):
                seen_owners.add(owner)
                try:
                    self._rpc.cache_clear(owner, self._name)
                except Exception:
                    logger.debug("Failed to clear on %s", owner)

    # ── Local RPC handlers ────────────────────────────────────────

    def handle_put(self, payload: dict) -> dict:
        """Handle a remote put request."""
        pid = self._strategy.get_partition_id(payload["key"])
        ttl = payload.get("ttl", self._default_ttl)
        key = payload["key"]
        value = payload["value"]

        if self._wal and self._wal.enabled:
            self._wal.log_put(pid, self._name, key, value, ttl)
        self._store.put(pid, self._name, key, value, ttl)
        self._enforce_lru(pid)
        if self._backup and self._backup.enabled:
            self._backup.replicate_put(pid, self._name, key, value, ttl)

        return {"success": True}

    def handle_get(self, payload: dict) -> dict:
        """Handle a remote get request."""
        pid = self._strategy.get_partition_id(payload["key"])
        value = self._store.get(pid, self._name, payload["key"])
        if value is not None:
            return {"value": value, "found": True}
        return {"value": None, "found": False}

    def handle_delete(self, payload: dict) -> dict:
        """Handle a remote delete request."""
        pid = self._strategy.get_partition_id(payload["key"])
        key = payload["key"]

        if self._wal and self._wal.enabled:
            self._wal.log_delete(pid, self._name, key)
        deleted = self._store.delete(pid, self._name, key)
        if deleted and self._backup and self._backup.enabled:
            self._backup.replicate_delete(pid, self._name, key)

        return {"deleted": deleted}

    def handle_exists(self, payload: dict) -> dict:
        """Handle a remote exists request."""
        pid = self._strategy.get_partition_id(payload["key"])
        return {"exists": self._store.exists(pid, self._name, payload["key"])}

    def handle_put_many(self, payload: dict) -> dict:
        """Handle a remote put_many request."""
        entries = payload.get("entries", {})
        ttl = payload.get("ttl", self._default_ttl)
        affected_pids: set[int] = set()

        for key, value in entries.items():
            pid = self._strategy.get_partition_id(key)
            affected_pids.add(pid)
            if self._wal and self._wal.enabled:
                self._wal.log_put(pid, self._name, key, value, ttl)
            self._store.put(pid, self._name, key, value, ttl)
            if self._backup and self._backup.enabled:
                self._backup.replicate_put(pid, self._name, key, value, ttl)

        for pid in affected_pids:
            self._enforce_lru(pid)

        return {"success": True, "count": len(entries)}

    def handle_get_many(self, payload: dict) -> dict:
        """Handle a remote get_many request."""
        keys = payload.get("keys", [])
        result: dict[str, Any] = {}
        for key in keys:
            pid = self._strategy.get_partition_id(key)
            value = self._store.get(pid, self._name, key)
            if value is not None:
                result[key] = value
        return {"entries": result}

    def handle_delete_many(self, payload: dict) -> dict:
        """Handle a remote delete_many request."""
        keys = payload.get("keys", [])
        deleted_count = 0
        for key in keys:
            pid = self._strategy.get_partition_id(key)
            if self._wal and self._wal.enabled:
                self._wal.log_delete(pid, self._name, key)
            if self._store.delete(pid, self._name, key):
                deleted_count += 1
                if self._backup and self._backup.enabled:
                    self._backup.replicate_delete(pid, self._name, key)
        return {"deleted_count": deleted_count}

    def handle_keys(self, payload: dict) -> dict:
        """Handle a remote keys request."""
        local_partitions = self._table.get_partitions_for_node(
            self._self_address
        )
        all_keys: list[str] = []
        for pid in local_partitions:
            all_keys.extend(self._store.keys(pid, self._name))
        return {"keys": all_keys}

    def handle_size(self, payload: dict) -> dict:
        """Handle a remote size request."""
        local_partitions = self._table.get_partitions_for_node(
            self._self_address
        )
        total = 0
        for pid in local_partitions:
            total += self._store.size(pid, self._name)
        return {"size": total}

    def handle_clear(self, payload: dict) -> dict:
        """Handle a remote clear request."""
        local_partitions = self._table.get_partitions_for_node(
            self._self_address
        )
        cleared = 0
        for pid in local_partitions:
            if self._wal and self._wal.enabled:
                self._wal.log_clear(pid, self._name)
            cleared += self._store.clear_collection(pid, self._name)
            if self._backup and self._backup.enabled:
                self._backup.replicate_clear(pid, self._name)
        return {"cleared": cleared}

    def __repr__(self) -> str:
        return (
            f"CacheCollection(name={self._name!r}, "
            f"ttl={self._default_ttl}, max_entries={self._max_entries})"
        )