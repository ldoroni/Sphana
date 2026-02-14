"""Cluster coordinator — manages partition assignment and membership.

The coordinator is the node with the **lowest address** (lexicographic).
It runs background loops for:
1. Heartbeat monitoring — detecting dead nodes.
2. Partition rebalancing — triggered on membership changes.
3. Partition table broadcasting — pushing updates to all members.

Non-coordinator nodes simply follow the partition table they receive.

Resiliency features:
- **Debounced rebalancing**: waits a stabilization period after
  membership changes before rebalancing, to avoid cascading rebalances
  during rolling deployments.
- **Quorum protection**: the coordinator only allows writes when it
  can see a majority of expected cluster members.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable

from ..models import RpcMessageType
from ..network.rpc_client import RpcClient
from ..partitioning.partition_rebalancer import PartitionRebalancer
from ..partitioning.partition_table import PartitionTable
from .node_registry import NodeRegistry

logger = logging.getLogger(__name__)

# How often the coordinator checks for dead nodes (seconds)
_HEARTBEAT_INTERVAL = 3.0
# A node is considered dead after this many seconds without heartbeat
_DEAD_NODE_TIMEOUT = 15.0
# Wait this long after membership change before rebalancing (seconds)
_REBALANCE_DEBOUNCE_SECONDS = 10.0


class Coordinator:
    """Cluster coordinator that manages partition ownership.

    Parameters:
        self_address: This node's ``host:port``.
        node_registry: Shared :class:`NodeRegistry`.
        partition_table: Shared :class:`PartitionTable`.
        rebalancer: :class:`PartitionRebalancer` instance.
        rpc_client: Shared :class:`RpcClient`.
        on_rebalance: Optional callback invoked after rebalance with
            ``(old_owners, new_owners, changes)`` — used by the main
            ``DistributedCache`` to trigger data migration.
        rebalance_debounce: Seconds to wait after a membership change
            before triggering a rebalance (default 10s).  Set to 0 to
            rebalance immediately.
        expected_cluster_size: If provided, quorum is computed against
            this value.  Otherwise, the largest cluster size ever seen
            is used.
    """

    def __init__(
        self,
        self_address: str,
        node_registry: NodeRegistry,
        partition_table: PartitionTable,
        rebalancer: PartitionRebalancer,
        rpc_client: RpcClient,
        on_rebalance: Callable[
            [dict[int, str], dict[int, str], dict[int, tuple[str | None, str]]],
            None,
        ]
        | None = None,
        rebalance_debounce: float = _REBALANCE_DEBOUNCE_SECONDS,
        expected_cluster_size: int | None = None,
    ) -> None:
        self._self_address = self_address
        self._registry = node_registry
        self._table = partition_table
        self._rebalancer = rebalancer
        self._rpc = rpc_client
        self._on_rebalance = on_rebalance

        self._running = False
        self._thread: threading.Thread | None = None
        self._rebalance_lock = threading.Lock()
        self._last_member_hash: int = 0

        # ── Debounced rebalancing ─────────────────────────────────
        self._rebalance_debounce = rebalance_debounce
        self._membership_changed_at: float | None = None
        self._pending_rebalance = False

        # ── Quorum tracking ───────────────────────────────────────
        self._expected_cluster_size = expected_cluster_size
        self._max_cluster_size_seen: int = 1
        self._has_quorum = True
        self._quorum_lock = threading.Lock()

        # ── Last rebalance timestamp (for metrics) ────────────────
        self._last_rebalance_at: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────

    @property
    def is_coordinator(self) -> bool:
        """True if this node is the current coordinator."""
        members = self._registry.get_active_addresses()
        if not members:
            return True
        return self._self_address == min(members)

    @property
    def has_quorum(self) -> bool:
        """True if a majority of expected cluster members are visible."""
        with self._quorum_lock:
            return self._has_quorum

    @property
    def last_rebalance_at(self) -> float:
        """Epoch timestamp of the last rebalance."""
        return self._last_rebalance_at

    def start(self) -> None:
        """Start the coordinator background loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="coordinator", daemon=True
        )
        self._thread.start()
        logger.info("Coordinator loop started on %s", self._self_address)

    def stop(self) -> None:
        """Stop the coordinator background loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ── Main loop ─────────────────────────────────────────────────

    def _loop(self) -> None:
        # Initial rebalance (immediate, no debounce on startup)
        self._do_rebalance_if_needed(force=True)

        while self._running:
            try:
                if self.is_coordinator:
                    self._check_heartbeats()
                    self._do_rebalance_if_needed()
                    self._broadcast_partition_table()
                self._update_quorum()
                self._send_heartbeats()
            except Exception:
                logger.exception("Coordinator loop error")
            time.sleep(_HEARTBEAT_INTERVAL)

    # ── Heartbeat ─────────────────────────────────────────────────

    def _send_heartbeats(self) -> None:
        """Send heartbeats to all known peers."""
        peers = [
            a
            for a in self._registry.get_active_addresses()
            if a != self._self_address
        ]
        for peer in peers:
            self._rpc.send_heartbeat_safe(
                peer,
                node_id=self._self_address,
                partition_table_version=self._table.version,
            )

    def _check_heartbeats(self) -> None:
        """Detect and remove dead nodes."""
        newly_dead = self._registry.mark_dead_nodes()
        if newly_dead:
            self._registry.remove_dead_nodes()

    # ── Quorum ────────────────────────────────────────────────────

    def _update_quorum(self) -> None:
        """Recompute whether we have a quorum of cluster members."""
        visible = len(self._registry.get_active_addresses())

        # Track the max cluster size we've ever seen
        if visible > self._max_cluster_size_seen:
            self._max_cluster_size_seen = visible

        expected = self._expected_cluster_size or self._max_cluster_size_seen

        # Single-node cluster always has quorum
        if expected <= 1:
            quorum_ok = True
        else:
            required = math.ceil(expected / 2)
            quorum_ok = visible >= required

        with self._quorum_lock:
            if self._has_quorum and not quorum_ok:
                logger.warning(
                    "Quorum LOST: only %d/%d nodes visible "
                    "(need %d for majority)",
                    visible,
                    expected,
                    math.ceil(expected / 2),
                )
            elif not self._has_quorum and quorum_ok:
                logger.info(
                    "Quorum RESTORED: %d/%d nodes visible", visible, expected
                )
            self._has_quorum = quorum_ok

    # ── Debounced Rebalancing ─────────────────────────────────────

    def _do_rebalance_if_needed(self, *, force: bool = False) -> None:
        """Rebalance with debouncing — wait for stabilization after
        membership changes before actually rebalancing.

        Args:
            force: If True, skip debounce and rebalance immediately.
        """
        members = self._registry.get_active_addresses()
        member_hash = hash(tuple(sorted(members)))

        if member_hash != self._last_member_hash:
            # Membership changed
            self._last_member_hash = member_hash
            self._membership_changed_at = time.time()
            self._pending_rebalance = True

        if not self._pending_rebalance:
            return

        if force or self._rebalance_debounce <= 0:
            self._pending_rebalance = False
            self._membership_changed_at = None
            self.rebalance()
            return

        # Wait for the debounce period
        assert self._membership_changed_at is not None
        elapsed = time.time() - self._membership_changed_at
        if elapsed >= self._rebalance_debounce:
            logger.info(
                "Rebalance debounce period (%.1fs) elapsed, proceeding",
                self._rebalance_debounce,
            )
            self._pending_rebalance = False
            self._membership_changed_at = None
            self.rebalance()

    def rebalance(self) -> None:
        """Force a partition rebalance now."""
        with self._rebalance_lock:
            members = self._registry.get_active_addresses()
            if not members:
                return

            old_owners = self._table.get_all_owners()
            new_owners, new_backups = self._rebalancer.compute_assignment(
                members
            )
            changes = self._rebalancer.compute_diff(old_owners, new_owners)

            if not changes:
                return

            self._table.assign(new_owners, new_backups)
            self._last_rebalance_at = time.time()

            logger.info(
                "Rebalanced: %d partition changes across %d nodes (v%d)",
                len(changes),
                len(members),
                self._table.version,
            )

            if self._on_rebalance:
                try:
                    self._on_rebalance(old_owners, new_owners, changes)
                except Exception:
                    logger.exception("on_rebalance callback failed")

    # ── Broadcast ─────────────────────────────────────────────────

    def _broadcast_partition_table(self) -> None:
        """Push the current partition table to all peers."""
        table_data = self._table.to_dict()
        peers = [
            a
            for a in self._registry.get_active_addresses()
            if a != self._self_address
        ]
        for peer in peers:
            try:
                self._rpc.send_partition_table_update(peer, table_data)
            except Exception:
                logger.debug("Failed to send partition table to %s", peer)

    # ── Handle incoming messages ──────────────────────────────────

    def handle_heartbeat(
        self, sender: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a heartbeat from a peer node."""
        self._registry.update_heartbeat(sender)

        # If sender has a newer partition table, request it
        remote_version = payload.get("partition_table_version", 0)
        return {
            "partition_table_version": self._table.version,
            "is_coordinator": self.is_coordinator,
            "needs_table_update": remote_version < self._table.version,
        }

    def handle_partition_table_update(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a partition table update from the coordinator."""
        table_data = payload.get("partition_table", {})
        self._table.from_dict(table_data)
        return {"version": self._table.version}

    def handle_node_join(self, address: str) -> dict[str, Any]:
        """Handle a new node joining the cluster."""
        self._registry.update_heartbeat(address)
        if self.is_coordinator:
            self._do_rebalance_if_needed()
        return {"accepted": True, "coordinator": self._self_address}

    def handle_node_leave(self, address: str) -> dict[str, Any]:
        """Handle a node gracefully leaving."""
        self._registry.update_from_provider(
            [a for a in self._registry.get_active_addresses() if a != address]
        )
        if self.is_coordinator:
            # On explicit leave, rebalance immediately (no debounce)
            self._do_rebalance_if_needed(force=True)
        return {"acknowledged": True}