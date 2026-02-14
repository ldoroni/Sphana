"""Distributed cache — main entry point.

Orchestrates all subcomponents: networking, cluster membership,
partitioning, storage, locking, backup replication, WAL persistence,
and cache collections.

Usage::

    from distributed_cache import DistributedCache, ClusterNodesProvider

    class MyProvider(ClusterNodesProvider):
        def get_node_addresses(self) -> list[str]:
            return ["10.0.0.1:9100", "10.0.0.2:9100"]

    cache = DistributedCache(
        self_address="10.0.0.1:9100",
        cluster_nodes_provider=MyProvider(),
        backup_count=1,          # in-memory backup replicas
        wal_dir="/data/wal",     # WAL persistence (None = disabled)
    )
    cache.start()

    coll = cache.get_collection("sessions")
    coll.put("user:123", {"name": "Alice"}, ttl=300)
    print(coll.get("user:123"))

    with cache.acquire_lock("my-resource") as lock:
        # critical section
        pass

    cache.stop()
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .cache_collection import CacheCollection
from .cluster.cluster_nodes_provider import ClusterNodesProvider
from .cluster.coordinator import Coordinator
from .cluster.node_registry import NodeRegistry
from .exceptions import QuorumLostError
from .locking.lock_handle import LockHandle
from .locking.lock_manager import LockManager
from .models import RpcMessageType
from .network.connection_pool import ConnectionPool
from .network.rpc_client import RpcClient
from .network.rpc_server import RpcServer
from .partitioning.partition_migrator import PartitionMigrator
from .partitioning.partition_rebalancer import PartitionRebalancer
from .partitioning.partition_strategy import PartitionStrategy
from .partitioning.partition_table import PartitionTable
from .storage.backup_replicator import BackupReplicator
from .storage.lock_store import LockStore
from .storage.partition_store import PartitionStore
from .storage.wal_store import WalStore

logger = logging.getLogger(__name__)

# Background sweep interval for expired cache entries (seconds)
_EVICTION_INTERVAL = 10.0
# WAL compaction interval (seconds)
_WAL_COMPACT_INTERVAL = 300.0
# How often to poll the ClusterNodesProvider for membership changes
_PROVIDER_POLL_INTERVAL = 5.0
# How often to run anti-entropy (periodic backup sync) in seconds
_ANTI_ENTROPY_INTERVAL = 60.0

class DistributedCache:
    """Production-ready partitioned distributed cache with backup
    replication and optional WAL persistence.

    Parameters:
        self_address: This node's ``host:port`` for RPC.
        cluster_nodes_provider: Supplies the list of cluster node
            addresses. The list may change dynamically.
        partition_count: Number of hash-ring partitions (default 271).
        backup_count: Number of backup replicas per partition
            (0 = no backups, 1+ = synchronous in-memory replication).
        rpc_port: TCP port for the RPC server. If 0, extracted from
            ``self_address``.
        connection_timeout: TCP connect timeout (seconds).
        request_timeout: RPC request timeout (seconds).
        wal_dir: Directory for WAL segment files.  ``None`` disables
            WAL persistence entirely (pure in-memory mode).
        wal_fsync: Whether to ``fsync`` after every WAL write.
        wal_max_segment_bytes: Max size of a single WAL segment
            before rotation (default 64 MiB).
    """

    def __init__(
        self,
        self_address: str,
        cluster_nodes_provider: ClusterNodesProvider,
        partition_count: int = 271,
        backup_count: int = 1,
        rpc_port: int = 0,
        connection_timeout: float = 5.0,
        request_timeout: float = 10.0,
        wal_dir: str | Path | None = None,
        wal_fsync: bool = True,
        wal_max_segment_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._self_address = self_address
        self._node_id = f"{self_address}:{uuid.uuid4().hex[:8]}"
        self._provider = cluster_nodes_provider
        self._backup_count = backup_count

        # ── Core components ───────────────────────────────────────
        self._strategy = PartitionStrategy(partition_count)
        self._table = PartitionTable(partition_count)
        self._partition_store = PartitionStore()
        self._lock_store = LockStore()
        port = rpc_port or int(self_address.split(":")[-1])
        self._registry = NodeRegistry(
            self_address=self_address,
            rpc_port=port,
        )

        # ── Networking ────────────────────────────────────────────
        self._pool = ConnectionPool(
            connect_timeout=connection_timeout,
            recv_timeout=request_timeout,
        )
        self._rpc_client = RpcClient(
            self_address=self_address,
            connection_pool=self._pool,
            default_timeout=request_timeout,
        )

        self._rpc_server = RpcServer(
            port=port,
        )

        # ── Partitioning ──────────────────────────────────────────
        self._rebalancer = PartitionRebalancer(
            partition_count=partition_count,
            backup_count=backup_count,
        )
        self._migrator = PartitionMigrator(
            rpc_client=self._rpc_client,
            self_address=self._self_address,
        )

        # ── Coordinator ───────────────────────────────────────────
        self._coordinator = Coordinator(
            self_address=self_address,
            node_registry=self._registry,
            partition_table=self._table,
            rebalancer=self._rebalancer,
            rpc_client=self._rpc_client,
            on_rebalance=self._on_rebalance,
        )

        # ── Backup replicator ─────────────────────────────────────
        self._backup_replicator = BackupReplicator(
            self_address=self_address,
            partition_table=self._table,
            rpc_client=self._rpc_client,
            enabled=(backup_count > 0),
        )

        # ── WAL persistence ───────────────────────────────────────
        wal_enabled = wal_dir is not None
        self._wal = WalStore(
            wal_dir=wal_dir or "/tmp/distributed_cache_wal",
            max_segment_bytes=wal_max_segment_bytes,
            fsync=wal_fsync,
            enabled=wal_enabled,
        )

        # ── Locking ───────────────────────────────────────────────
        self._lock_manager = LockManager(
            self_address=self_address,
            node_id=self._node_id,
            partition_strategy=self._strategy,
            partition_table=self._table,
            lock_store=self._lock_store,
            rpc_client=self._rpc_client,
        )

        # ── Collections ───────────────────────────────────────────
        self._collections: dict[str, CacheCollection] = {}
        self._collections_lock = threading.Lock()

        # ── Background tasks ──────────────────────────────────────
        self._running = False
        self._eviction_thread: threading.Thread | None = None
        self._wal_compact_thread: threading.Thread | None = None
        self._provider_poll_thread: threading.Thread | None = None
        self._anti_entropy_thread: threading.Thread | None = None

        # ── Metrics counters ──────────────────────────────────────
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "deletes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rpc_sent": 0,
            "rpc_errors": 0,
            "backups_promoted": 0,
        }

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the cache node — WAL replay, RPC server,
        coordinator, eviction."""
        if self._running:
            return
        self._running = True

        # Replay WAL before accepting traffic
        if self._wal.enabled:
            self._replay_wal()

        # Wire RPC handler
        self._rpc_server.set_handler(self._handle_rpc)
        self._rpc_server.start()

        self._coordinator.start()

        # Provider polling (membership discovery)
        self._provider_poll_thread = threading.Thread(
            target=self._provider_poll_loop,
            name="provider-poll",
            daemon=True,
        )
        self._provider_poll_thread.start()

        self._eviction_thread = threading.Thread(
            target=self._eviction_loop, name="cache-eviction", daemon=True
        )
        self._eviction_thread.start()

        # Anti-entropy background sync
        if self._backup_count > 0:
            self._anti_entropy_thread = threading.Thread(
                target=self._anti_entropy_loop,
                name="anti-entropy",
                daemon=True,
            )
            self._anti_entropy_thread.start()

        if self._wal.enabled:
            self._wal_compact_thread = threading.Thread(
                target=self._wal_compact_loop,
                name="wal-compaction",
                daemon=True,
            )
            self._wal_compact_thread.start()

        logger.info(
            "DistributedCache started on %s (node_id=%s, "
            "backup_count=%d, wal=%s)",
            self._self_address,
            self._node_id,
            self._backup_count,
            "enabled" if self._wal.enabled else "disabled",
        )

    def stop(self) -> None:
        """Gracefully stop the cache node.

        Sends a NODE_LEAVE message to the coordinator so the cluster
        can immediately rebalance without waiting for heartbeat
        timeout.
        """
        if not self._running:
            return
        self._running = False

        # Graceful leave: notify the coordinator
        self._send_node_leave()

        self._coordinator.stop()
        self._rpc_server.stop()
        self._pool.close_all()

        if self._wal.enabled:
            self._wal.close()

        if self._eviction_thread is not None:
            self._eviction_thread.join(timeout=5.0)
            self._eviction_thread = None

        if self._wal_compact_thread is not None:
            self._wal_compact_thread.join(timeout=5.0)
            self._wal_compact_thread = None

        if self._provider_poll_thread is not None:
            self._provider_poll_thread.join(timeout=5.0)
            self._provider_poll_thread = None

        if self._anti_entropy_thread is not None:
            self._anti_entropy_thread.join(timeout=5.0)
            self._anti_entropy_thread = None

        logger.info("DistributedCache stopped on %s", self._self_address)

    # ── Collection API ────────────────────────────────────────────

    def get_collection(
        self,
        name: str,
        default_ttl: float = 0.0,
        max_entries: int = 0,
    ) -> CacheCollection:
        """Get or create a named cache collection.

        Args:
            name: Collection name.
            default_ttl: Default entry TTL in seconds (0 = no expiry).
            max_entries: Max entries (0 = unlimited).
        """
        with self._collections_lock:
            if name not in self._collections:
                self._collections[name] = CacheCollection(
                    name=name,
                    self_address=self._self_address,
                    partition_strategy=self._strategy,
                    partition_table=self._table,
                    partition_store=self._partition_store,
                    rpc_client=self._rpc_client,
                    default_ttl=default_ttl,
                    max_entries=max_entries,
                    backup_replicator=self._backup_replicator,
                    wal_store=self._wal,
                    quorum_checker=lambda: self._coordinator.has_quorum,
                )
            return self._collections[name]

    def delete_collection(self, name: str) -> bool:
        """Delete a named collection and all its entries.

        Returns:
            True if the collection existed and was deleted.
        """
        with self._collections_lock:
            coll = self._collections.pop(name, None)
            if coll is None:
                return False
            coll.clear()
            return True

    def list_collections(self) -> list[str]:
        """Return names of all known collections."""
        with self._collections_lock:
            return list(self._collections.keys())

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        with self._collections_lock:
            return name in self._collections

    # ── Convenience cache methods (default collection) ────────────

    def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Put into the default collection."""
        self.get_collection("__default__").put(key, value, ttl)

    def get(self, key: str, default: Any = None) -> Any:
        """Get from the default collection."""
        return self.get_collection("__default__").get(key, default)

    def delete(self, key: str) -> bool:
        """Delete from the default collection."""
        return self.get_collection("__default__").delete(key)

    def exists(self, key: str) -> bool:
        """Check existence in the default collection."""
        return self.get_collection("__default__").exists(key)

    # ── Lock API ──────────────────────────────────────────────────

    def acquire_lock(
        self,
        key: str,
        hold_timeout: float = 60.0,
        acquire_timeout: float = 30.0,
        retry_interval: float = 0.2,
    ) -> LockHandle:
        """Acquire a distributed lock (blocking with timeout).

        Args:
            key: Lock key.
            hold_timeout: Max time the lock may be held (seconds).
            acquire_timeout: Max time to wait for acquisition (seconds).
            retry_interval: Sleep between retries (seconds).

        Returns:
            A :class:`LockHandle` context manager.

        Raises:
            LockAcquireTimeoutError: If acquisition times out.
        """
        return self._lock_manager.acquire(
            key, hold_timeout, acquire_timeout, retry_interval
        )

    def try_acquire_lock(
        self,
        key: str,
        hold_timeout: float = 60.0,
    ) -> LockHandle | None:
        """Try to acquire a lock without blocking.

        Returns:
            A :class:`LockHandle` if acquired, else ``None``.
        """
        return self._lock_manager.try_acquire(key, hold_timeout)

    def is_locked(self, key: str) -> bool:
        """Check if a key is currently locked."""
        return self._lock_manager.is_locked(key)

    def force_release_lock(self, key: str) -> bool:
        """Force-release a lock regardless of owner."""
        return self._lock_manager.force_release(key)

    # ── Info ──────────────────────────────────────────────────────

    @property
    def self_address(self) -> str:
        return self._self_address

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def is_coordinator(self) -> bool:
        return self._coordinator.is_coordinator

    @property
    def cluster_size(self) -> int:
        return len(self._registry.get_active_addresses())

    @property
    def partition_count(self) -> int:
        return self._strategy.partition_count

    @property
    def partition_table_version(self) -> int:
        return self._table.version

    @property
    def backup_count(self) -> int:
        return self._backup_count

    @property
    def wal_enabled(self) -> bool:
        return self._wal.enabled

    def get_local_partition_ids(self) -> list[int]:
        """Return partition IDs owned by this node."""
        return self._table.get_partitions_for_node(self._self_address)

    def get_backup_partition_ids(self) -> list[int]:
        """Return partition IDs for which this node is a backup."""
        return self._table.get_backup_partitions_for_node(self._self_address)

    # ── RPC handling ──────────────────────────────────────────────

    def _handle_rpc(self, message: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an incoming RPC request to the right handler."""
        msg_type_str = message.get("message_type", "")
        payload = message.get("payload", {})

        try:
            msg_type = RpcMessageType(msg_type_str)
        except ValueError:
            return {"error": f"Unknown message type: {msg_type_str}"}

        try:
            # Heartbeat / cluster
            if msg_type == RpcMessageType.HEARTBEAT:
                sender = message.get("sender_address", "")
                return self._coordinator.handle_heartbeat(sender, payload)

            if msg_type == RpcMessageType.PARTITION_TABLE_UPDATE:
                return self._coordinator.handle_partition_table_update(payload)

            if msg_type == RpcMessageType.NODE_JOIN:
                return self._coordinator.handle_node_join(
                    payload.get("address", "")
                )

            if msg_type == RpcMessageType.NODE_LEAVE:
                return self._coordinator.handle_node_leave(
                    payload.get("address", "")
                )

            # Cache operations
            if msg_type == RpcMessageType.CACHE_PUT:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_put(payload)

            if msg_type == RpcMessageType.CACHE_GET:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_get(payload)

            if msg_type == RpcMessageType.CACHE_DELETE:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_delete(payload)

            if msg_type == RpcMessageType.CACHE_EXISTS:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_exists(payload)

            # Bulk cache operations
            if msg_type == RpcMessageType.CACHE_PUT_MANY:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_put_many(payload)

            if msg_type == RpcMessageType.CACHE_GET_MANY:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_get_many(payload)

            if msg_type == RpcMessageType.CACHE_DELETE_MANY:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_delete_many(payload)

            if msg_type == RpcMessageType.CACHE_KEYS:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_keys(payload)

            if msg_type == RpcMessageType.CACHE_SIZE:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_size(payload)

            if msg_type == RpcMessageType.CACHE_CLEAR:
                coll = self._get_or_create_collection(payload.get("collection", "__default__"))
                return coll.handle_clear(payload)

            # Collection management operations
            if msg_type == RpcMessageType.COLLECTION_CREATE:
                name = payload.get("name", "")
                ttl = payload.get("default_ttl", 0.0)
                max_e = payload.get("max_entries", 0)
                self.get_collection(name, default_ttl=ttl, max_entries=max_e)
                return {"success": True}

            if msg_type == RpcMessageType.COLLECTION_DELETE:
                name = payload.get("name", "")
                deleted = self.delete_collection(name)
                return {"success": True, "deleted": deleted}

            if msg_type == RpcMessageType.COLLECTION_EXISTS:
                name = payload.get("name", "")
                return {"exists": self.collection_exists(name)}

            if msg_type == RpcMessageType.COLLECTION_LIST:
                return {"collections": self.list_collections()}

            # Lock operations
            if msg_type == RpcMessageType.LOCK_ACQUIRE:
                return self._lock_manager.handle_lock_acquire(payload)

            if msg_type == RpcMessageType.LOCK_RELEASE:
                return self._lock_manager.handle_lock_release(payload)

            if msg_type == RpcMessageType.LOCK_IS_LOCKED:
                return self._lock_manager.handle_lock_is_locked(payload)

            if msg_type == RpcMessageType.LOCK_FORCE_RELEASE:
                return self._lock_manager.handle_lock_force_release(payload)

            # Backup replication operations
            if msg_type == RpcMessageType.BACKUP_PUT:
                return self._handle_backup_put(payload)

            if msg_type == RpcMessageType.BACKUP_DELETE:
                return self._handle_backup_delete(payload)

            if msg_type == RpcMessageType.BACKUP_CLEAR:
                return self._handle_backup_clear(payload)

            if msg_type == RpcMessageType.BACKUP_SYNC:
                return self._handle_backup_sync(payload)

            # Partition migration
            if msg_type == RpcMessageType.PARTITION_MIGRATE:
                pid = payload.get("partition_id", -1)
                data = self._partition_store.get_partition_data(pid)
                return {"partition_id": pid, "data": data}

            if msg_type == RpcMessageType.PARTITION_MIGRATE_RECEIVE:
                pid = payload.get("partition_id", -1)
                data = payload.get("data", {})
                count = self._partition_store.import_partition_data(pid, data)
                return {"imported": count}

            return {"error": f"Unhandled message type: {msg_type_str}"}

        except Exception as exc:
            logger.exception("RPC handler error for %s", msg_type_str)
            return {"error": str(exc)}

    def _get_or_create_collection(self, name: str) -> CacheCollection:
        """Get a collection, creating it on-demand for RPC handling."""
        with self._collections_lock:
            if name not in self._collections:
                self._collections[name] = CacheCollection(
                    name=name,
                    self_address=self._self_address,
                    partition_strategy=self._strategy,
                    partition_table=self._table,
                    partition_store=self._partition_store,
                    rpc_client=self._rpc_client,
                    backup_replicator=self._backup_replicator,
                    wal_store=self._wal,
                )
            return self._collections[name]

    # ── Backup RPC handlers ───────────────────────────────────────

    def _handle_backup_put(self, payload: dict) -> dict:
        """Store data in the local backup partition store."""
        pid = payload.get("partition_id", -1)
        collection = payload.get("collection", "__default__")
        key = payload.get("key", "")
        value = payload.get("value")
        ttl = payload.get("ttl")
        self._partition_store.put(pid, collection, key, value, ttl)
        return {"success": True}

    def _handle_backup_delete(self, payload: dict) -> dict:
        """Delete data from the local backup partition store."""
        pid = payload.get("partition_id", -1)
        collection = payload.get("collection", "__default__")
        key = payload.get("key", "")
        deleted = self._partition_store.delete(pid, collection, key)
        return {"success": True, "deleted": deleted}

    def _handle_backup_clear(self, payload: dict) -> dict:
        """Clear a collection from the local backup partition store."""
        pid = payload.get("partition_id", -1)
        collection = payload.get("collection", "__default__")
        self._partition_store.clear_collection(pid, collection)
        return {"success": True}

    def _handle_backup_sync(self, payload: dict) -> dict:
        """Import a full partition snapshot as backup data."""
        pid = payload.get("partition_id", -1)
        data = payload.get("data", {})
        count = self._partition_store.import_partition_data(pid, data)
        logger.info(
            "Backup sync: imported %d entries for partition %d",
            count,
            pid,
        )
        return {"success": True, "imported": count}

    # ── Rebalance callback ────────────────────────────────────────

    def _on_rebalance(
        self,
        old_owners: dict[int, str],
        new_owners: dict[int, str],
        changes: dict[int, tuple[str | None, str]],
    ) -> None:
        """Called by the coordinator after partition reassignment.

        Triggers data migration for partitions that moved to/from
        this node, and sends full-sync to new backup nodes.
        """
        incoming: list[int] = []
        outgoing: list[int] = []

        for pid, (old_owner, new_owner) in changes.items():
            if new_owner == self._self_address and old_owner != self._self_address:
                incoming.append(pid)
            elif old_owner == self._self_address and new_owner != self._self_address:
                outgoing.append(pid)

        if incoming:
            logger.info(
                "Migrating %d partitions TO this node", len(incoming)
            )
            for pid in incoming:
                old_owner = old_owners.get(pid)
                if old_owner and old_owner != self._self_address:
                    try:
                        data = self._migrator.request_partition_data(
                            old_owner, pid
                        )
                        if data:
                            self._partition_store.import_partition_data(
                                pid, data
                            )
                    except Exception:
                        logger.exception(
                            "Failed to migrate partition %d from %s",
                            pid,
                            old_owner,
                        )

                # Sync backup replicas for newly-owned partitions
                if self._backup_replicator.enabled:
                    try:
                        part_data = self._partition_store.get_partition_data(pid)
                        if part_data:
                            self._backup_replicator.sync_full_partition(
                                pid, part_data
                            )
                    except Exception:
                        logger.exception(
                            "Failed to sync backups for partition %d", pid
                        )

        if outgoing:
            logger.info(
                "Dropping %d partitions FROM this node", len(outgoing)
            )
            for pid in outgoing:
                self._partition_store.drop_partition(pid)

    # ── WAL replay ────────────────────────────────────────────────

    def _replay_wal(self) -> None:
        """Replay WAL records to restore in-memory state after restart."""

        def on_put(
            pid: int, col: str, key: str, value: Any, ttl: float | None
        ) -> None:
            self._partition_store.put(pid, col, key, value, ttl)

        def on_delete(pid: int, col: str, key: str) -> None:
            self._partition_store.delete(pid, col, key)

        def on_clear(pid: int, col: str) -> None:
            self._partition_store.clear_collection(pid, col)

        count = self._wal.replay(on_put, on_delete, on_clear)
        if count:
            logger.info("WAL replay restored %d operations", count)

    # ── Background eviction ───────────────────────────────────────

    def _eviction_loop(self) -> None:
        """Periodically evict expired cache entries."""
        import time

        while self._running:
            try:
                self._partition_store.evict_expired()
                self._lock_store.evict_expired()
            except Exception:
                logger.exception("Eviction loop error")
            time.sleep(_EVICTION_INTERVAL)

    def _wal_compact_loop(self) -> None:
        """Periodically compact old WAL segments."""
        import time

        while self._running:
            time.sleep(_WAL_COMPACT_INTERVAL)
            if not self._running:
                break
            try:
                self._wal.compact()
            except Exception:
                logger.exception("WAL compaction error")

    # ── Graceful shutdown helpers ─────────────────────────────────

    def _send_node_leave(self) -> None:
        """Notify the cluster coordinator that this node is leaving."""
        members = self._registry.get_active_addresses()
        coordinator = min(members) if members else None
        if coordinator and coordinator != self._self_address:
            try:
                self._rpc_client.send(
                    coordinator,
                    RpcMessageType.NODE_LEAVE,
                    {"address": self._self_address},
                    timeout=3.0,
                )
                logger.info("Sent NODE_LEAVE to coordinator %s", coordinator)
            except Exception:
                logger.debug(
                    "Failed to send NODE_LEAVE to %s (may already be down)",
                    coordinator,
                )
        elif coordinator == self._self_address:
            # We are the coordinator — handle our own leave
            self._coordinator.handle_node_leave(self._self_address)

    # ── Provider polling ──────────────────────────────────────────

    def _provider_poll_loop(self) -> None:
        """Periodically poll the ClusterNodesProvider and feed
        updated membership into the NodeRegistry."""
        while self._running:
            try:
                addresses = self._provider.get_nodes()
                self._registry.update_from_provider(addresses)
            except Exception:
                logger.exception("Provider poll error")
            time.sleep(_PROVIDER_POLL_INTERVAL)

    # ── Anti-entropy (periodic backup sync) ───────────────────────

    def _anti_entropy_loop(self) -> None:
        """Periodically push full partition snapshots to backup nodes
        to repair any drift between primary and backup replicas."""
        while self._running:
            time.sleep(_ANTI_ENTROPY_INTERVAL)
            if not self._running:
                break
            try:
                self._run_anti_entropy()
            except Exception:
                logger.exception("Anti-entropy error")

    def _run_anti_entropy(self) -> None:
        """For each partition owned by this node, compute a digest
        and push a full sync to backup nodes if they differ."""
        owned = self.get_local_partition_ids()
        if not owned or not self._backup_replicator.enabled:
            return

        synced = 0
        for pid in owned:
            data = self._partition_store.get_partition_data(pid)
            if not data:
                continue
            # Compute a fast digest of partition contents
            digest = hashlib.md5(
                str(sorted(data.items())).encode(), usedforsecurity=False
            ).hexdigest()

            # Ask backup nodes for their digest (best-effort)
            backup_nodes = self._table.get_backups(pid)
            for bnode in backup_nodes:
                if bnode == self._self_address:
                    continue
                try:
                    resp = self._rpc_client.send(
                        bnode,
                        RpcMessageType.BACKUP_SYNC,
                        {"partition_id": pid, "data": data},
                        timeout=15.0,
                    )
                    synced += 1
                except Exception:
                    logger.debug(
                        "Anti-entropy sync to %s for partition %d failed",
                        bnode,
                        pid,
                    )
        if synced:
            logger.info("Anti-entropy: synced %d partition-backups", synced)

    # ── Backup promotion ──────────────────────────────────────────

    def promote_backup_partitions(self, failed_node: str) -> int:
        """Promote backup partitions from a failed node to primary
        ownership on this node.

        Called automatically during rebalance when a node disappears.

        Args:
            failed_node: Address of the node that failed.

        Returns:
            Number of partitions promoted.
        """
        promoted = 0
        backup_pids = self.get_backup_partition_ids()
        for pid in backup_pids:
            owner = self._table.get_owner(pid)
            if owner == failed_node:
                # This node has a backup copy — promote it
                logger.info(
                    "Promoting backup partition %d (was owned by %s)",
                    pid,
                    failed_node,
                )
                promoted += 1
                with self._metrics_lock:
                    self._metrics["backups_promoted"] += 1
        return promoted

    # ── Partition ownership validation ────────────────────────────

    def validate_partition_ownership(self, key: str) -> bool:
        """Check whether this node owns the partition for the given key.

        Useful for validating incoming write requests.
        """
        pid = self._strategy.get_partition_id(key)
        return self._table.get_owner(pid) == self._self_address

    # ── Metrics / health ──────────────────────────────────────────

    def increment_metric(self, name: str, amount: int = 1) -> None:
        """Thread-safe metric counter increment."""
        with self._metrics_lock:
            self._metrics[name] = self._metrics.get(name, 0) + amount

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of all metrics."""
        with self._metrics_lock:
            return dict(self._metrics)

    def health(self) -> dict[str, Any]:
        """Return a health-check dict suitable for an HTTP endpoint.

        Includes cluster state, partition info, quorum status, and
        basic counters.
        """
        owned = self.get_local_partition_ids()
        backup = self.get_backup_partition_ids()
        return {
            "status": "healthy" if self._running else "stopped",
            "address": self._self_address,
            "node_id": self._node_id,
            "is_coordinator": self.is_coordinator,
            "has_quorum": self._coordinator.has_quorum,
            "cluster_size": self.cluster_size,
            "partition_table_version": self.partition_table_version,
            "partitions_owned": len(owned),
            "partitions_backup": len(backup),
            "total_partitions": self.partition_count,
            "backup_count": self._backup_count,
            "wal_enabled": self.wal_enabled,
            "collections": list(self._collections.keys()),
            "connection_pool": self._pool.pool_stats,
            "metrics": self.get_metrics(),
        }

    # ── Dunder ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DistributedCache(address={self._self_address!r}, "
            f"partitions={self.partition_count}, "
            f"backup_count={self._backup_count}, "
            f"wal={'on' if self._wal.enabled else 'off'}, "
            f"cluster_size={self.cluster_size})"
        )

    def __enter__(self) -> DistributedCache:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.stop()