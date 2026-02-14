"""Node registry — tracks known cluster members and their state.

The registry is updated periodically by polling the
:class:`ClusterNodesProvider` and by receiving heartbeats from peers.
It provides the ordered member list used for coordinator election
(oldest-member strategy).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from ..models import NodeInfo, NodeState

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Thread-safe registry of known cluster nodes.

    Parameters:
        self_address:
            The ``host:port`` address of the local node.
        rpc_port:
            The TCP port used for RPC communication.
        heartbeat_timeout:
            Seconds after which a node with no heartbeat is considered dead.
    """

    def __init__(
        self,
        self_address: str,
        rpc_port: int = 4322,
        heartbeat_timeout: float = 15.0,
    ) -> None:
        self._lock = threading.Lock()
        self._heartbeat_timeout = heartbeat_timeout

        # Build the local node entry
        self._self_node_id = f"{self_address}-{uuid.uuid4().hex[:8]}"
        self._self_address = self_address

        now = time.time()
        self._self_node = NodeInfo(
            address=self_address,
            node_id=self._self_node_id,
            state=NodeState.ACTIVE,
            join_timestamp=now,
            last_heartbeat=now,
            rpc_port=rpc_port,
        )

        # All known nodes keyed by address
        self._nodes: dict[str, NodeInfo] = {self_address: self._self_node}

    # ── Properties ────────────────────────────────────────────────

    @property
    def self_address(self) -> str:
        return self._self_address

    @property
    def self_node_id(self) -> str:
        return self._self_node_id

    @property
    def self_node(self) -> NodeInfo:
        return self._self_node

    # ── Query Methods ─────────────────────────────────────────────

    def get_active_nodes(self) -> list[NodeInfo]:
        """Return all nodes currently in ACTIVE state, sorted by join_timestamp."""
        with self._lock:
            return sorted(
                [n for n in self._nodes.values() if n.state == NodeState.ACTIVE],
                key=lambda n: n.join_timestamp,
            )

    def get_all_nodes(self) -> list[NodeInfo]:
        """Return all tracked nodes regardless of state."""
        with self._lock:
            return list(self._nodes.values())

    def get_node(self, address: str) -> NodeInfo | None:
        """Return the node info for a given address, or None."""
        with self._lock:
            return self._nodes.get(address)

    def get_active_addresses(self) -> list[str]:
        """Return sorted list of active node addresses."""
        return [n.address for n in self.get_active_nodes()]

    def is_self(self, address: str) -> bool:
        """Check if the given address is the local node."""
        return address == self._self_address

    def get_oldest_member_address(self) -> str | None:
        """Return the address of the oldest active member (coordinator candidate)."""
        active = self.get_active_nodes()
        if not active:
            return None
        return active[0].address

    def is_coordinator(self) -> bool:
        """Check if the local node should be the coordinator (oldest member)."""
        return self.get_oldest_member_address() == self._self_address

    def node_count(self) -> int:
        """Return the number of active nodes."""
        with self._lock:
            return sum(1 for n in self._nodes.values() if n.state == NodeState.ACTIVE)

    # ── Mutation Methods ──────────────────────────────────────────

    def update_from_provider(self, provider_addresses: list[str]) -> tuple[list[str], list[str]]:
        """Reconcile the registry with a fresh list from the provider.

        Args:
            provider_addresses: Current list of addresses from the
                :class:`ClusterNodesProvider`.

        Returns:
            A tuple of ``(joined_addresses, left_addresses)`` representing
            the membership diff since the last update.
        """
        now = time.time()
        provider_set = set(provider_addresses)
        joined: list[str] = []
        left: list[str] = []

        with self._lock:
            current_addresses = set(self._nodes.keys())

            # New nodes
            for addr in provider_set - current_addresses:
                self._nodes[addr] = NodeInfo(
                    address=addr,
                    node_id=f"{addr}-unknown",
                    state=NodeState.ACTIVE,
                    join_timestamp=now,
                    last_heartbeat=now,
                )
                joined.append(addr)
                logger.info("Node joined: %s", addr)

            # Nodes that disappeared from the provider
            for addr in current_addresses - provider_set:
                if addr == self._self_address:
                    continue  # Never remove self
                node = self._nodes[addr]
                if node.state == NodeState.ACTIVE:
                    node.state = NodeState.DEAD
                    left.append(addr)
                    logger.info("Node left: %s", addr)

            # Re-activate nodes that reappeared
            for addr in provider_set & current_addresses:
                node = self._nodes[addr]
                if node.state == NodeState.DEAD:
                    node.state = NodeState.ACTIVE
                    node.last_heartbeat = now
                    joined.append(addr)
                    logger.info("Node re-joined: %s", addr)

        return joined, left

    def update_heartbeat(self, address: str, node_id: str | None = None) -> None:
        """Record a heartbeat from a peer node."""
        now = time.time()
        with self._lock:
            node = self._nodes.get(address)
            if node is None:
                # First time seeing this node via heartbeat
                self._nodes[address] = NodeInfo(
                    address=address,
                    node_id=node_id or f"{address}-unknown",
                    state=NodeState.ACTIVE,
                    join_timestamp=now,
                    last_heartbeat=now,
                )
            else:
                node.last_heartbeat = now
                if node_id and node.node_id.endswith("-unknown"):
                    node.node_id = node_id
                if node.state == NodeState.DEAD:
                    node.state = NodeState.ACTIVE
                    logger.info("Node revived via heartbeat: %s", address)

    def mark_dead_nodes(self) -> list[str]:
        """Mark nodes as DEAD if their heartbeat has expired.

        Returns:
            List of addresses that were newly marked dead.
        """
        now = time.time()
        newly_dead: list[str] = []

        with self._lock:
            for node in self._nodes.values():
                if node.address == self._self_address:
                    # Always keep self alive
                    node.last_heartbeat = now
                    continue
                if (
                    node.state == NodeState.ACTIVE
                    and (now - node.last_heartbeat) > self._heartbeat_timeout
                ):
                    node.state = NodeState.DEAD
                    newly_dead.append(node.address)
                    logger.warning(
                        "Node %s marked dead (no heartbeat for %.1fs)",
                        node.address,
                        now - node.last_heartbeat,
                    )

        return newly_dead

    def remove_dead_nodes(self) -> list[str]:
        """Remove all DEAD nodes from the registry.

        Returns:
            List of removed addresses.
        """
        with self._lock:
            dead_addrs = [
                addr
                for addr, node in self._nodes.items()
                if node.state == NodeState.DEAD and addr != self._self_address
            ]
            for addr in dead_addrs:
                del self._nodes[addr]
            return dead_addrs

    def touch_self(self) -> None:
        """Update the local node's heartbeat timestamp."""
        self._self_node.last_heartbeat = time.time()