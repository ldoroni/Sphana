"""Partition-aware distributed task router.

Routes tasks to the node that owns the partition for a given routing
key, using the same partition table and strategy as the distributed
cache.  If the target is the local node, the registered handler is
invoked directly (no RPC overhead).

Self-healing: when a node dies the coordinator detects the missing
heartbeat, the rebalancer reassigns partitions to surviving nodes,
and subsequent ``submit()`` calls are automatically routed to the
new owner — no manual intervention required.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from ..exceptions import RpcConnectionError, RpcError, RpcTimeoutError
from ..models import RpcMessageType
from ..network.rpc_client import RpcClient
from ..partitioning.partition_strategy import PartitionStrategy
from ..partitioning.partition_table import PartitionTable
from .task_result import TaskResult, TaskStatus

logger = logging.getLogger(__name__)

# Type alias: handler(routing_key, message) -> response_dict | None
TaskHandler = Callable[[str, dict[str, Any]], dict[str, Any] | None]


class DistributedTaskRouter:
    """Route arbitrary task messages to the partition-owner node.

    Parameters:
        self_address: This node's ``host:port``.
        partition_strategy: Maps routing keys → partition IDs.
        partition_table: Maps partition IDs → owner node addresses.
        rpc_client: For sending messages to remote nodes.
    """

    def __init__(
        self,
        self_address: str,
        partition_strategy: PartitionStrategy,
        partition_table: PartitionTable,
        rpc_client: RpcClient,
    ) -> None:
        self._self_address = self_address
        self._strategy = partition_strategy
        self._table = partition_table
        self._rpc_client = rpc_client
        self._handlers: dict[str, TaskHandler] = {}
        self._handlers_lock = threading.Lock()

    # ── Listener registration ─────────────────────────────────────

    def listen(self, topic: str, handler: TaskHandler) -> None:
        """Register a local handler for a topic.

        When a ``TASK_SUBMIT`` message arrives (either locally or via
        RPC) for the given topic, *handler* is invoked with
        ``(routing_key, message_dict)`` and must return a response
        dict (or ``None``).

        Args:
            topic: Topic identifier (e.g. ``"shard.ingest_document"``).
            handler: Callable ``(routing_key, message) → dict | None``.
        """
        with self._handlers_lock:
            if topic in self._handlers:
                logger.warning("Overwriting handler for topic: %s", topic)
            self._handlers[topic] = handler
            logger.info("Registered task handler for topic: %s", topic)

    # ── Submit (routed to partition owner) ────────────────────────

    def submit(
        self,
        routing_key: str,
        topic: str,
        message: dict[str, Any],
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> TaskResult:
        """Submit a task to the node that owns the partition for
        *routing_key*.

        If the owner is this node the local handler is called
        directly.  Otherwise the message is forwarded via RPC.

        Args:
            routing_key: Key used to determine the target partition
                (e.g. shard name).
            topic: Topic identifier for handler dispatch.
            message: Payload dict.
            timeout: RPC timeout in seconds (ignored for local).
            max_retries: Number of RPC retry attempts on transient
                connection / timeout errors.

        Returns:
            A :class:`TaskResult` with the handler response.

        Raises:
            ValueError: If a local dispatch has no registered handler.
            RpcConnectionError: If the remote node is unreachable
                after all retries.
            RpcTimeoutError: If the remote node times out after all
                retries.
            RpcError: If the remote node returns a logical error.
        """
        partition_id = self._strategy.get_partition_id(routing_key)
        owner = self._table.get_owner(partition_id)

        if owner is None:
            return TaskResult.failure(
                error=f"No owner for partition {partition_id} "
                      f"(routing_key={routing_key!r}). "
                      "Cluster may be rebalancing.",
                node_address=self._self_address,
            )

        if owner == self._self_address:
            return self._invoke_local(topic, routing_key, message)

        return self._invoke_remote(
            owner, topic, routing_key, message,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ── Broadcast (to all nodes) ──────────────────────────────────

    def submit_to_all(
        self,
        topic: str,
        message: dict[str, Any],
        *,
        timeout: float = 120.0,
    ) -> dict[str, TaskResult]:
        """Broadcast a task to every node in the cluster.

        Invokes the handler on each unique node that owns at least
        one partition (including this node).  Useful for operations
        that must run on every pod, such as ``create_index`` or
        ``delete_index``.

        Args:
            topic: Topic identifier.
            message: Payload dict.
            timeout: RPC timeout per node.

        Returns:
            A dict mapping ``node_address → TaskResult``.
        """
        owners_map = self._table.get_all_owners()
        unique_nodes: set[str] = set(owners_map.values())
        results: dict[str, TaskResult] = {}

        for node_address in unique_nodes:
            if node_address == self._self_address:
                results[node_address] = self._invoke_local(
                    topic, "", message
                )
            else:
                results[node_address] = self._invoke_remote(
                    node_address, topic, "", message,
                    timeout=timeout,
                    max_retries=1,
                )

        return results

    # ── RPC handler (called by RpcServer on incoming TASK_SUBMIT) ─

    def handle_task_submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming ``TASK_SUBMIT`` RPC message.

        Called by :meth:`DistributedCache._handle_rpc` when the
        message type is ``TASK_SUBMIT``.

        Args:
            payload: Must contain ``topic``, ``routing_key``, and
                ``message`` keys.

        Returns:
            A dict with ``status``, ``response``, and optionally
            ``error`` — matching :class:`TaskResult` fields.
        """
        topic = payload.get("topic", "")
        routing_key = payload.get("routing_key", "")
        message = payload.get("message", {})

        result = self._invoke_local(topic, routing_key, message)

        return {
            "status": result.status.value,
            "response": result.response,
            "error": result.error,
        }

    # ── Internal dispatch ─────────────────────────────────────────

    def _invoke_local(
        self,
        topic: str,
        routing_key: str,
        message: dict[str, Any],
    ) -> TaskResult:
        """Invoke the locally registered handler for *topic*."""
        with self._handlers_lock:
            handler = self._handlers.get(topic)

        if handler is None:
            return TaskResult.failure(
                error=f"No handler registered for topic: {topic}",
                node_address=self._self_address,
            )

        try:
            response = handler(routing_key, message)
            return TaskResult.success(
                response=response or {},
                node_address=self._self_address,
            )
        except Exception as exc:
            logger.exception(
                "Task handler error for topic=%s routing_key=%s",
                topic,
                routing_key,
            )
            return TaskResult.failure(
                error=str(exc),
                node_address=self._self_address,
            )

    def _invoke_remote(
        self,
        target: str,
        topic: str,
        routing_key: str,
        message: dict[str, Any],
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> TaskResult:
        """Forward a task to a remote node via RPC."""
        payload = {
            "topic": topic,
            "routing_key": routing_key,
            "message": message,
        }

        try:
            rpc_response = self._rpc_client.send_with_retry(
                target,
                RpcMessageType.TASK_SUBMIT,
                payload,
                timeout=timeout,
                max_retries=max_retries,
            )

            # Parse the remote TaskResult from the RPC response
            resp_payload = rpc_response.payload
            status_str = resp_payload.get("status", TaskStatus.FAILED.value)
            error = resp_payload.get("error")
            response = resp_payload.get("response", {})

            return TaskResult(
                status=TaskStatus(status_str),
                response=response,
                error=error,
                node_address=target,
            )

        except (RpcConnectionError, RpcTimeoutError) as exc:
            logger.error(
                "Task submit to %s failed after retries: %s", target, exc
            )
            return TaskResult.failure(
                error=f"Node {target} unreachable: {exc}",
                node_address=target,
            )
        except RpcError as exc:
            logger.error(
                "Task submit to %s returned error: %s", target, exc
            )
            return TaskResult.failure(
                error=f"Remote error from {target}: {exc}",
                node_address=target,
            )