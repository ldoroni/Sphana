"""Partition migrator — moves data between nodes during rebalancing.

When the coordinator detects partition ownership changes, it instructs
nodes to migrate data for the affected partitions.  This module
handles the RPC-based migration protocol.
"""

from __future__ import annotations

import logging
from typing import Any

from ..network.rpc_client import RpcClient
from ..models import RpcMessageType

logger = logging.getLogger(__name__)

class PartitionMigrator:
    """Handles migration of partition data between nodes.

    Parameters:
        rpc_client: The RPC client used to communicate with peers.
        self_address: The address of the local node.
    """

    def __init__(self, rpc_client: RpcClient, self_address: str) -> None:
        self._rpc_client = rpc_client
        self._self_address = self_address

    def request_partition_data(
        self,
        source_address: str,
        partition_id: int,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Request all data for a partition from a source node.

        Args:
            source_address: The node currently holding the data.
            partition_id: The partition to migrate.
            collection: Optional collection name filter.  If None,
                migrates all collections.

        Returns:
            Dict containing the partition data:
            ``{"entries": {collection: {key: value, ...}, ...}}``
        """
        try:
            response = self._rpc_client.send(
                source_address,
                RpcMessageType.PARTITION_MIGRATE,
                {
                    "partition_id": partition_id,
                    "collection": collection,
                    "target_address": self._self_address,
                },
                timeout=30.0,
            )
            return response.payload
        except Exception:
            logger.exception(
                "Failed to request partition %d data from %s",
                partition_id,
                source_address,
            )
            return {"entries": {}}

    def send_partition_data(
        self,
        target_address: str,
        partition_id: int,
        data: dict[str, dict[str, Any]],
    ) -> bool:
        """Push partition data to a target node.

        Args:
            target_address: The new owner node.
            partition_id: The partition being migrated.
            data: Dict of ``{collection: {key: value, ...}, ...}``.

        Returns:
            True if migration was accepted, False otherwise.
        """
        try:
            self._rpc_client.send(
                target_address,
                RpcMessageType.PARTITION_MIGRATE_RECEIVE,
                {
                    "partition_id": partition_id,
                    "entries": data,
                    "source_address": self._self_address,
                },
                timeout=30.0,
            )
            logger.info(
                "Migrated partition %d to %s (%d collections)",
                partition_id,
                target_address,
                len(data),
            )
            return True
        except Exception:
            logger.exception(
                "Failed to send partition %d data to %s",
                partition_id,
                target_address,
            )
            return False

    def migrate_partitions(
        self,
        changes: dict[int, tuple[str | None, str]],
        get_partition_data: Any,
    ) -> int:
        """Execute a batch of partition migrations.

        This method is called by the coordinator after a rebalance.
        For each partition that changed ownership, if the old owner
        is the local node, we push the data to the new owner.

        Args:
            changes: Dict of partition_id → (old_owner, new_owner).
            get_partition_data: Callable(partition_id) → dict of
                {collection: {key: value}} data for that partition.

        Returns:
            Number of successfully migrated partitions.
        """
        migrated = 0
        for pid, (old_owner, new_owner) in changes.items():
            if old_owner != self._self_address:
                continue  # Not our responsibility
            if new_owner == self._self_address:
                continue  # No-op

            data = get_partition_data(pid)
            if not data:
                migrated += 1
                continue  # Empty partition, nothing to move

            if self.send_partition_data(new_owner, pid, data):
                migrated += 1

        return migrated