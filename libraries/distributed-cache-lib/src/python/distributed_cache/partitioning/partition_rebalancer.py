"""Partition rebalancer — computes partition assignments for a set of nodes.

The rebalancer distributes partitions as evenly as possible across
active nodes.  It also assigns backup replicas when the backup count
is > 0 and enough nodes are available.

This module is stateless — it takes the current member list and
partition count and returns a new assignment.  The coordinator
calls this whenever membership changes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

class PartitionRebalancer:
    """Computes a balanced partition → node assignment.

    Parameters:
        partition_count: Total number of partitions (fixed).
        backup_count: Number of backup copies per partition (0 = no backups).
    """

    def __init__(
        self,
        partition_count: int,
        backup_count: int = 1,
    ) -> None:
        self._partition_count = partition_count
        self._backup_count = backup_count

    def compute_assignment(
        self,
        active_nodes: list[str],
    ) -> tuple[dict[int, str], dict[int, list[str]]]:
        """Compute a balanced partition assignment.

        Distributes partitions round-robin across the sorted list of
        active nodes.  Backup replicas are assigned to the *next*
        nodes in the sorted order (never the same node as the owner).

        Args:
            active_nodes: List of active node addresses.

        Returns:
            A tuple of ``(owners, backups)`` where:
            - ``owners``: dict mapping partition_id → owner address
            - ``backups``: dict mapping partition_id → list of backup addresses
        """
        if not active_nodes:
            logger.warning("No active nodes — returning empty assignment")
            return {}, {}

        sorted_nodes = sorted(active_nodes)
        node_count = len(sorted_nodes)

        owners: dict[int, str] = {}
        backups: dict[int, list[str]] = {}

        for pid in range(self._partition_count):
            # Round-robin owner
            owner_idx = pid % node_count
            owners[pid] = sorted_nodes[owner_idx]

            # Assign backups to subsequent nodes (skip the owner)
            if self._backup_count > 0 and node_count > 1:
                partition_backups: list[str] = []
                for b in range(1, self._backup_count + 1):
                    backup_idx = (owner_idx + b) % node_count
                    if backup_idx != owner_idx:
                        partition_backups.append(sorted_nodes[backup_idx])
                if partition_backups:
                    backups[pid] = partition_backups

        logger.info(
            "Rebalanced %d partitions across %d nodes (backup_count=%d)",
            self._partition_count,
            node_count,
            self._backup_count,
        )

        return owners, backups

    def compute_diff(
        self,
        old_owners: dict[int, str],
        new_owners: dict[int, str],
    ) -> dict[int, tuple[str | None, str]]:
        """Compute which partitions changed ownership.

        Returns:
            Dict mapping partition_id → (old_owner, new_owner) for
            partitions that changed owner.  ``old_owner`` is None if
            the partition was previously unassigned.
        """
        changes: dict[int, tuple[str | None, str]] = {}
        for pid, new_owner in new_owners.items():
            old_owner = old_owners.get(pid)
            if old_owner != new_owner:
                changes[pid] = (old_owner, new_owner)
        return changes