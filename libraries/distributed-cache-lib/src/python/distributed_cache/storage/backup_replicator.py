"""Backup replicator — synchronous in-memory replication to backup nodes.

When a primary node stores data, it also sends the same operation to
all backup nodes for that partition via RPC.  This ensures that if
the primary crashes, the backup node already has the data in memory
and can be promoted instantly with zero data loss.

Replication is **synchronous** for writes (put/delete) to guarantee
consistency.  If a backup node is unreachable, the write still
succeeds on the primary — backup failures are logged but do not
block the caller.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import RpcMessageType
from ..network.rpc_client import RpcClient
from ..partitioning.partition_table import PartitionTable

logger = logging.getLogger(__name__)


class BackupReplicator:
    """Replicates cache mutations to backup nodes.

    Parameters:
        self_address: This node's ``host:port``.
        partition_table: Maps partitions to primary + backup owners.
        rpc_client: For sending replication RPCs.
        enabled: Whether backup replication is active.
    """

    def __init__(
        self,
        self_address: str,
        partition_table: PartitionTable,
        rpc_client: RpcClient,
        enabled: bool = True,
    ) -> None:
        self._self_address = self_address
        self._table = partition_table
        self._rpc = rpc_client
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Replication methods ───────────────────────────────────────

    def replicate_put(
        self,
        partition_id: int,
        collection: str,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        """Replicate a put operation to all backup nodes for this partition.

        Called by the primary owner after a successful local put.
        Failures on individual backup nodes are logged but do not
        raise — the primary write is always considered successful.
        """
        if not self._enabled:
            return

        backups = self._table.get_backups(partition_id)
        if not backups:
            return

        payload = {
            "partition_id": partition_id,
            "collection": collection,
            "key": key,
            "value": value,
            "ttl": ttl,
        }

        for backup_addr in backups:
            if backup_addr == self._self_address:
                continue
            try:
                self._rpc.send(
                    backup_addr,
                    RpcMessageType.BACKUP_PUT,
                    payload,
                )
            except Exception:
                logger.warning(
                    "Failed to replicate PUT to backup %s for "
                    "partition %d key '%s'",
                    backup_addr,
                    partition_id,
                    key,
                )

    def replicate_delete(
        self,
        partition_id: int,
        collection: str,
        key: str,
    ) -> None:
        """Replicate a delete operation to all backup nodes."""
        if not self._enabled:
            return

        backups = self._table.get_backups(partition_id)
        if not backups:
            return

        payload = {
            "partition_id": partition_id,
            "collection": collection,
            "key": key,
        }

        for backup_addr in backups:
            if backup_addr == self._self_address:
                continue
            try:
                self._rpc.send(
                    backup_addr,
                    RpcMessageType.BACKUP_DELETE,
                    payload,
                )
            except Exception:
                logger.warning(
                    "Failed to replicate DELETE to backup %s for "
                    "partition %d key '%s'",
                    backup_addr,
                    partition_id,
                    key,
                )

    def replicate_clear(
        self,
        partition_id: int,
        collection: str,
    ) -> None:
        """Replicate a clear-collection operation to all backup nodes."""
        if not self._enabled:
            return

        backups = self._table.get_backups(partition_id)
        if not backups:
            return

        payload = {
            "partition_id": partition_id,
            "collection": collection,
        }

        for backup_addr in backups:
            if backup_addr == self._self_address:
                continue
            try:
                self._rpc.send(
                    backup_addr,
                    RpcMessageType.BACKUP_CLEAR,
                    payload,
                )
            except Exception:
                logger.warning(
                    "Failed to replicate CLEAR to backup %s for "
                    "partition %d collection '%s'",
                    backup_addr,
                    partition_id,
                    collection,
                )

    def sync_full_partition(
        self,
        partition_id: int,
        data: dict[str, dict[str, Any]],
    ) -> None:
        """Send a full partition snapshot to all backup nodes.

        Used during rebalance when a new backup is assigned.

        Args:
            partition_id: The partition ID.
            data: ``{collection: {key: value, ...}, ...}``
        """
        if not self._enabled:
            return

        backups = self._table.get_backups(partition_id)
        if not backups:
            return

        payload = {
            "partition_id": partition_id,
            "data": data,
        }

        for backup_addr in backups:
            if backup_addr == self._self_address:
                continue
            try:
                self._rpc.send(
                    backup_addr,
                    RpcMessageType.BACKUP_SYNC,
                    payload,
                )
                logger.info(
                    "Full sync of partition %d sent to backup %s",
                    partition_id,
                    backup_addr,
                )
            except Exception:
                logger.warning(
                    "Failed to full-sync partition %d to backup %s",
                    partition_id,
                    backup_addr,
                )