"""High-level RPC client built on top of the connection pool.

Provides typed convenience methods for sending cache, lock, and
cluster RPC messages to peer nodes and awaiting responses.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from typing import Any

from ..exceptions import RpcConnectionError, RpcError, RpcTimeoutError
from ..models import RpcMessageType, RpcResponse
from .connection_pool import ConnectionPool

logger = logging.getLogger(__name__)

class RpcClient:
    """Client for sending RPC requests to peer nodes.

    Parameters:
        self_address: The ``host:port`` of the local node (used as
            the ``sender_address`` in outgoing messages).
        connection_pool: Shared :class:`ConnectionPool` instance.
        default_timeout: Default timeout in seconds for RPC calls.
    """

    def __init__(
        self,
        self_address: str,
        connection_pool: ConnectionPool,
        default_timeout: float = 10.0,
    ) -> None:
        self._self_address = self_address
        self._pool = connection_pool
        self._default_timeout = default_timeout

    # ── Generic Send ──────────────────────────────────────────────

    def send(
        self,
        target: str,
        message_type: RpcMessageType,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> RpcResponse:
        """Send an RPC message and return the parsed response.

        Args:
            target: ``host:port`` of the destination node.
            message_type: The RPC message type.
            payload: Message-specific payload dict.
            timeout: Override the default recv timeout.

        Returns:
            An :class:`RpcResponse` parsed from the raw response.

        Raises:
            RpcError: If the remote node returns an error response.
            RpcConnectionError: On connection failure.
            RpcTimeoutError: On timeout.
        """
        request_id = uuid.uuid4().hex
        message: dict[str, Any] = {
            "message_type": message_type.value,
            "request_id": request_id,
            "sender_address": self._self_address,
            "payload": payload or {},
            "timestamp": time.time(),
        }

        raw = self._pool.send_and_receive(
            target, message, timeout=timeout or self._default_timeout
        )

        response = RpcResponse(
            request_id=raw.get("request_id", request_id),
            success=raw.get("success", False),
            payload=raw.get("payload", {}),
            error=raw.get("error"),
        )

        if not response.success:
            raise RpcError(target, reason=response.error or "Unknown error")

        return response

    # ── Cache Operations ──────────────────────────────────────────

    def cache_put(
        self,
        target: str,
        collection: str,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_PUT, {
            "collection": collection,
            "key": key,
            "value": value,
            "ttl": ttl,
        })

    def cache_get(self, target: str, collection: str, key: str) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_GET, {
            "collection": collection,
            "key": key,
        })

    def cache_delete(self, target: str, collection: str, key: str) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_DELETE, {
            "collection": collection,
            "key": key,
        })

    def cache_exists(self, target: str, collection: str, key: str) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_EXISTS, {
            "collection": collection,
            "key": key,
        })

    def cache_keys(self, target: str, collection: str) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_KEYS, {
            "collection": collection,
        })

    def cache_size(self, target: str, collection: str) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_SIZE, {
            "collection": collection,
        })

    def cache_clear(self, target: str, collection: str) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_CLEAR, {
            "collection": collection,
        })

    def cache_put_many(
        self,
        target: str,
        collection: str,
        entries: dict[str, Any],
        ttl: float | None = None,
    ) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_PUT_MANY, {
            "collection": collection,
            "entries": entries,
            "ttl": ttl,
        })

    def cache_get_many(
        self, target: str, collection: str, keys: list[str]
    ) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_GET_MANY, {
            "collection": collection,
            "keys": keys,
        })

    def cache_delete_many(
        self, target: str, collection: str, keys: list[str]
    ) -> RpcResponse:
        return self.send(target, RpcMessageType.CACHE_DELETE_MANY, {
            "collection": collection,
            "keys": keys,
        })

    # ── Lock Operations ───────────────────────────────────────────

    def lock_acquire(
        self,
        target: str,
        key: str,
        owner_id: str,
        hold_timeout: float = 60.0,
    ) -> RpcResponse:
        return self.send(target, RpcMessageType.LOCK_ACQUIRE, {
            "key": key,
            "owner_id": owner_id,
            "hold_timeout": hold_timeout,
        })

    def lock_release(
        self, target: str, key: str, owner_id: str, fencing_token: int = 0
    ) -> RpcResponse:
        return self.send(target, RpcMessageType.LOCK_RELEASE, {
            "key": key,
            "owner_id": owner_id,
            "fencing_token": fencing_token,
        })

    def lock_is_locked(self, target: str, key: str) -> RpcResponse:
        return self.send(target, RpcMessageType.LOCK_IS_LOCKED, {"key": key})

    def lock_force_release(self, target: str, key: str) -> RpcResponse:
        return self.send(target, RpcMessageType.LOCK_FORCE_RELEASE, {"key": key})

    # ── Cluster Operations ────────────────────────────────────────

    def heartbeat(
        self, target: str, node_id: str, partition_table_version: int = 0
    ) -> RpcResponse:
        return self.send(target, RpcMessageType.HEARTBEAT, {
            "node_id": node_id,
            "partition_table_version": partition_table_version,
        })

    def send_partition_table_update(
        self, target: str, partition_table: dict[str, Any]
    ) -> RpcResponse:
        return self.send(target, RpcMessageType.PARTITION_TABLE_UPDATE, {
            "partition_table": partition_table,
        })

    # ── Retry with Backoff ────────────────────────────────────────

    def send_with_retry(
        self,
        target: str,
        message_type: RpcMessageType,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 5.0,
    ) -> RpcResponse:
        """Send an RPC message with exponential backoff retry.

        Retries on ``RpcConnectionError`` and ``RpcTimeoutError``.
        Does **not** retry on ``RpcError`` (remote explicitly returned
        an error) as that indicates a logical failure.

        Args:
            target: ``host:port`` of the destination node.
            message_type: The RPC message type.
            payload: Message-specific payload dict.
            timeout: Override the default recv timeout.
            max_retries: Maximum retry attempts (0 = no retries).
            base_delay: Initial delay between retries in seconds.
            max_delay: Cap on the exponential backoff delay.

        Returns:
            An :class:`RpcResponse` parsed from the raw response.

        Raises:
            RpcError: If the remote node returns an error response.
            RpcConnectionError: If all retries are exhausted.
            RpcTimeoutError: If all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                return self.send(target, message_type, payload, timeout)
            except (RpcConnectionError, RpcTimeoutError) as exc:
                last_error = exc
                if attempt < max_retries:
                    delay = min(
                        base_delay * (2 ** attempt) + random.uniform(0, 0.3),
                        max_delay,
                    )
                    logger.warning(
                        "RPC to %s failed (attempt %d/%d), retrying in %.2fs: %s",
                        target,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                        exc,
                    )
                    time.sleep(delay)

        assert last_error is not None
        raise last_error

    # ── Safe helpers ──────────────────────────────────────────────

    def send_heartbeat_safe(
        self, target: str, node_id: str, partition_table_version: int = 0
    ) -> bool:
        """Send a heartbeat, returning True on success, False on failure.

        Swallows connection errors so the caller doesn't need try/except.
        """
        try:
            self.heartbeat(target, node_id, partition_table_version)
            return True
        except (RpcError, RpcConnectionError, OSError):
            return False