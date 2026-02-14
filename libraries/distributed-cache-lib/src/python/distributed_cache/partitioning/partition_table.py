"""Partition table — maps each partition ID to its owner node.

The table is versioned; every time the coordinator reassigns
partitions, the version is bumped.  Nodes use the version to
detect stale routing information.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

class PartitionTable:
    """Thread-safe partition-to-node ownership mapping.

    Parameters:
        partition_count: Total number of partitions.
    """

    def __init__(self, partition_count: int) -> None:
        self._lock = threading.Lock()
        self._partition_count = partition_count
        self._version = 0

        # partition_id → owner address
        self._owners: dict[int, str] = {}
        # partition_id → list of backup addresses
        self._backups: dict[int, list[str]] = {}

    # ── Properties ────────────────────────────────────────────────

    @property
    def version(self) -> int:
        return self._version

    @property
    def partition_count(self) -> int:
        return self._partition_count

    # ── Query ─────────────────────────────────────────────────────

    def get_owner(self, partition_id: int) -> str | None:
        """Return the owner address for a partition, or None."""
        with self._lock:
            return self._owners.get(partition_id)

    def get_backups(self, partition_id: int) -> list[str]:
        """Return the list of backup node addresses for a partition."""
        with self._lock:
            return list(self._backups.get(partition_id, []))

    def get_all_owners(self) -> dict[int, str]:
        """Return a snapshot of the full ownership map."""
        with self._lock:
            return dict(self._owners)

    def get_partitions_for_node(self, address: str) -> list[int]:
        """Return all partition IDs owned by a given node."""
        with self._lock:
            return [
                pid for pid, owner in self._owners.items()
                if owner == address
            ]

    def get_backup_partitions_for_node(self, address: str) -> list[int]:
        """Return all partition IDs for which a node is a backup."""
        with self._lock:
            return [
                pid for pid, backups in self._backups.items()
                if address in backups
            ]

    def is_owner(self, partition_id: int, address: str) -> bool:
        with self._lock:
            return self._owners.get(partition_id) == address

    def is_assigned(self, partition_id: int) -> bool:
        with self._lock:
            return partition_id in self._owners

    # ── Mutation ──────────────────────────────────────────────────

    def assign(
        self,
        owners: dict[int, str],
        backups: dict[int, list[str]] | None = None,
        version: int | None = None,
    ) -> None:
        """Replace the entire partition table.

        Args:
            owners: Mapping of partition_id → owner address.
            backups: Optional mapping of partition_id → backup addresses.
            version: Explicit version number.  If None, auto-increments.
        """
        with self._lock:
            self._owners = dict(owners)
            self._backups = {k: list(v) for k, v in (backups or {}).items()}
            if version is not None:
                self._version = version
            else:
                self._version += 1
            logger.info(
                "Partition table updated to version %d (%d partitions assigned)",
                self._version,
                len(self._owners),
            )

    def set_owner(self, partition_id: int, address: str) -> None:
        """Set the owner of a single partition (increments version)."""
        with self._lock:
            self._owners[partition_id] = address
            self._version += 1

    def set_backups(self, partition_id: int, addresses: list[str]) -> None:
        """Set the backup nodes of a single partition."""
        with self._lock:
            self._backups[partition_id] = list(addresses)

    def remove_node(self, address: str) -> list[int]:
        """Remove a node from all partition assignments.

        Returns:
            List of partition IDs that lost their primary owner.
        """
        orphaned: list[int] = []
        with self._lock:
            # Remove from owners
            to_remove = [
                pid for pid, owner in self._owners.items()
                if owner == address
            ]
            for pid in to_remove:
                del self._owners[pid]
                orphaned.append(pid)

            # Remove from backups
            for pid in list(self._backups.keys()):
                self._backups[pid] = [
                    a for a in self._backups[pid] if a != address
                ]
                if not self._backups[pid]:
                    del self._backups[pid]

            if orphaned:
                self._version += 1

        return orphaned

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for transmission over RPC."""
        with self._lock:
            return {
                "version": self._version,
                "partition_count": self._partition_count,
                "owners": {str(k): v for k, v in self._owners.items()},
                "backups": {
                    str(k): v for k, v in self._backups.items()
                },
            }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Deserialize from a dict received over RPC.

        Only applies if the incoming version is newer than the current.
        """
        incoming_version = data.get("version", 0)
        with self._lock:
            if incoming_version <= self._version:
                return
            self._version = incoming_version
            self._partition_count = data.get(
                "partition_count", self._partition_count
            )
            self._owners = {
                int(k): v for k, v in data.get("owners", {}).items()
            }
            self._backups = {
                int(k): v for k, v in data.get("backups", {}).items()
            }
            logger.info(
                "Partition table updated from remote (version %d)",
                self._version,
            )