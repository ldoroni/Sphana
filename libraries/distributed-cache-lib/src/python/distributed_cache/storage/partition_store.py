"""Local in-memory partition store.

Each node holds data only for the partitions it owns (or backs up).
Data is organized as: partition_id → collection → key → CacheEntry.
TTL eviction is handled lazily on reads and periodically by a
background sweeper.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ..models import CacheEntry

logger = logging.getLogger(__name__)

class PartitionStore:
    """Thread-safe in-memory storage for partitioned cache data.

    Data is stored in a nested dict:
    ``{partition_id: {collection: {key: CacheEntry}}}``
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # partition_id → collection → key → CacheEntry
        self._data: dict[int, dict[str, dict[str, CacheEntry]]] = {}

    # ── Single-key operations ─────────────────────────────────────

    def put(
        self,
        partition_id: int,
        collection: str,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        now = time.time()
        entry = CacheEntry(
            key=key,
            value=value,
            collection=collection,
            created_at=now,
            updated_at=now,
            expires_at=(now + ttl) if ttl else None,
            last_accessed_at=now,
        )
        with self._lock:
            part = self._data.setdefault(partition_id, {})
            coll = part.setdefault(collection, {})
            existing = coll.get(key)
            if existing is not None:
                entry.version = existing.version + 1
                entry.created_at = existing.created_at
            coll[key] = entry

    def get(
        self, partition_id: int, collection: str, key: str
    ) -> Any | None:
        with self._lock:
            entry = self._get_entry(partition_id, collection, key)
            if entry is None:
                return None
            if entry.is_expired:
                self._delete_entry(partition_id, collection, key)
                return None
            entry.touch()
            return entry.value

    def delete(self, partition_id: int, collection: str, key: str) -> bool:
        with self._lock:
            return self._delete_entry(partition_id, collection, key)

    def exists(self, partition_id: int, collection: str, key: str) -> bool:
        with self._lock:
            entry = self._get_entry(partition_id, collection, key)
            if entry is None:
                return False
            if entry.is_expired:
                self._delete_entry(partition_id, collection, key)
                return False
            return True

    # ── Bulk operations ───────────────────────────────────────────

    def put_many(
        self,
        partition_id: int,
        collection: str,
        entries: dict[str, Any],
        ttl: float | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            part = self._data.setdefault(partition_id, {})
            coll = part.setdefault(collection, {})
            for key, value in entries.items():
                existing = coll.get(key)
                version = (existing.version + 1) if existing else 1
                created = existing.created_at if existing else now
                coll[key] = CacheEntry(
                    key=key,
                    value=value,
                    collection=collection,
                    created_at=created,
                    updated_at=now,
                    expires_at=(now + ttl) if ttl else None,
                    last_accessed_at=now,
                    version=version,
                )

    def get_many(
        self, partition_id: int, collection: str, keys: list[str]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        with self._lock:
            for key in keys:
                entry = self._get_entry(partition_id, collection, key)
                if entry is not None and not entry.is_expired:
                    entry.touch()
                    result[key] = entry.value
                elif entry is not None:
                    self._delete_entry(partition_id, collection, key)
        return result

    def delete_many(
        self, partition_id: int, collection: str, keys: list[str]
    ) -> int:
        count = 0
        with self._lock:
            for key in keys:
                if self._delete_entry(partition_id, collection, key):
                    count += 1
        return count

    # ── Collection operations ─────────────────────────────────────

    def keys(self, partition_id: int, collection: str) -> list[str]:
        with self._lock:
            coll = self._data.get(partition_id, {}).get(collection, {})
            return [k for k, v in coll.items() if not v.is_expired]

    def size(self, partition_id: int, collection: str) -> int:
        with self._lock:
            coll = self._data.get(partition_id, {}).get(collection, {})
            return sum(1 for v in coll.values() if not v.is_expired)

    def clear_collection(self, partition_id: int, collection: str) -> int:
        with self._lock:
            coll = self._data.get(partition_id, {}).get(collection, {})
            count = len(coll)
            coll.clear()
            return count

    def collections(self, partition_id: int) -> list[str]:
        with self._lock:
            return list(self._data.get(partition_id, {}).keys())

    # ── Partition-level operations ────────────────────────────────

    def get_partition_data(
        self, partition_id: int
    ) -> dict[str, dict[str, Any]]:
        """Return all data for a partition (for migration).

        Returns:
            ``{collection: {key: value, ...}, ...}``
        """
        with self._lock:
            part = self._data.get(partition_id, {})
            result: dict[str, dict[str, Any]] = {}
            for coll_name, coll in part.items():
                entries: dict[str, Any] = {}
                for key, entry in coll.items():
                    if not entry.is_expired:
                        entries[key] = entry.value
                if entries:
                    result[coll_name] = entries
            return result

    def import_partition_data(
        self,
        partition_id: int,
        data: dict[str, dict[str, Any]],
    ) -> int:
        """Import migrated data for a partition.

        Args:
            partition_id: The partition ID.
            data: ``{collection: {key: value, ...}, ...}``

        Returns:
            Total number of entries imported.
        """
        count = 0
        now = time.time()
        with self._lock:
            part = self._data.setdefault(partition_id, {})
            for coll_name, entries in data.items():
                coll = part.setdefault(coll_name, {})
                for key, value in entries.items():
                    coll[key] = CacheEntry(
                        key=key,
                        value=value,
                        collection=coll_name,
                        created_at=now,
                        updated_at=now,
                        last_accessed_at=now,
                    )
                    count += 1
        return count

    def drop_partition(self, partition_id: int) -> int:
        """Remove all data for a partition. Returns entry count removed."""
        with self._lock:
            part = self._data.pop(partition_id, {})
            return sum(len(c) for c in part.values())

    def owned_partitions(self) -> list[int]:
        """List all partition IDs with local data."""
        with self._lock:
            return list(self._data.keys())

    # ── LRU eviction ──────────────────────────────────────────────

    def evict_lru(
        self, partition_id: int, collection: str, max_entries: int
    ) -> int:
        """Evict least-recently-used entries from a collection until
        the entry count is at or below ``max_entries``.

        Args:
            partition_id: The partition ID.
            collection: The collection name.
            max_entries: Maximum allowed entries (must be > 0).

        Returns:
            Number of entries evicted.
        """
        if max_entries <= 0:
            return 0
        evicted = 0
        with self._lock:
            coll = self._data.get(partition_id, {}).get(collection, {})
            current_size = len(coll)
            if current_size <= max_entries:
                return 0
            # Sort by last_accessed_at ascending (oldest first)
            sorted_keys = sorted(
                coll.keys(),
                key=lambda k: coll[k].last_accessed_at,
            )
            to_evict = current_size - max_entries
            for k in sorted_keys[:to_evict]:
                del coll[k]
                evicted += 1
        if evicted:
            logger.debug(
                "LRU-evicted %d entries from partition %d collection %s",
                evicted,
                partition_id,
                collection,
            )
        return evicted

    # ── TTL eviction ──────────────────────────────────────────────

    def evict_expired(self) -> int:
        """Remove all expired entries across all partitions.

        Returns:
            Number of entries evicted.
        """
        evicted = 0
        with self._lock:
            for part in self._data.values():
                for coll in part.values():
                    expired_keys = [
                        k for k, v in coll.items() if v.is_expired
                    ]
                    for k in expired_keys:
                        del coll[k]
                        evicted += 1
        if evicted:
            logger.debug("Evicted %d expired entries", evicted)
        return evicted

    # ── Private helpers ───────────────────────────────────────────

    def _get_entry(
        self, partition_id: int, collection: str, key: str
    ) -> CacheEntry | None:
        """Get an entry without locking (caller must hold lock)."""
        return self._data.get(partition_id, {}).get(collection, {}).get(key)

    def _delete_entry(
        self, partition_id: int, collection: str, key: str
    ) -> bool:
        """Delete an entry without locking (caller must hold lock)."""
        coll = self._data.get(partition_id, {}).get(collection, {})
        if key in coll:
            del coll[key]
            return True
        return False