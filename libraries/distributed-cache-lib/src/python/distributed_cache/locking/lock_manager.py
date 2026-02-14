"""Distributed lock manager — routes lock operations to partition owners.

Lock keys are hashed to partitions just like cache keys. The lock
request is forwarded to the partition owner (which may be local or
remote). The local :class:`LockStore` handles the actual state.
"""

from __future__ import annotations

import logging
import time
import uuid

from ..exceptions import LockAcquireTimeoutError
from ..models import LockResult, RpcMessageType
from ..network.rpc_client import RpcClient
from ..partitioning.partition_strategy import PartitionStrategy
from ..partitioning.partition_table import PartitionTable
from ..storage.lock_store import LockStore
from .lock_handle import LockHandle

logger = logging.getLogger(__name__)

class LockManager:
    """Routes lock acquire/release to the correct partition owner.

    Parameters:
        self_address: This node's ``host:port``.
        partition_strategy: Hashes keys to partition IDs.
        partition_table: Maps partitions to owner addresses.
        lock_store: Local lock state store.
        rpc_client: For forwarding to remote owners.
    """

    def __init__(
        self,
        self_address: str,
        node_id: str,
        partition_strategy: PartitionStrategy,
        partition_table: PartitionTable,
        lock_store: LockStore,
        rpc_client: RpcClient,
    ) -> None:
        self._self_address = self_address
        self._node_id = node_id
        self._strategy = partition_strategy
        self._table = partition_table
        self._lock_store = lock_store
        self._rpc = rpc_client

    # ── Public API ────────────────────────────────────────────────

    def acquire(
        self,
        key: str,
        hold_timeout: float = 60.0,
        acquire_timeout: float = 30.0,
        retry_interval: float = 0.2,
    ) -> LockHandle:
        """Acquire a distributed lock with retry.

        Args:
            key: The lock key.
            hold_timeout: Max seconds the lock may be held once acquired.
            acquire_timeout: Max seconds to wait for the lock.
            retry_interval: Seconds between retry attempts.

        Returns:
            A :class:`LockHandle` for the acquired lock.

        Raises:
            LockTimeoutError: If the lock cannot be acquired in time.
        """
        owner_id = f"{self._node_id}:{uuid.uuid4().hex[:8]}"
        deadline = time.monotonic() + acquire_timeout

        while True:
            result = self._try_acquire(key, owner_id, hold_timeout)
            if result.acquired:
                return LockHandle(
                    key=key,
                    owner_id=owner_id,
                    fencing_token=result.fencing_token,
                    lock_manager=self,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockAcquireTimeoutError(
                    key=key,
                    timeout=acquire_timeout,
                    current_owner=result.owner_id,
                )

            time.sleep(min(retry_interval, remaining))

    def try_acquire(
        self,
        key: str,
        hold_timeout: float = 60.0,
    ) -> LockHandle | None:
        """Try to acquire a lock without waiting.

        Returns:
            A :class:`LockHandle` if acquired, or None.
        """
        owner_id = f"{self._node_id}:{uuid.uuid4().hex[:8]}"
        result = self._try_acquire(key, owner_id, hold_timeout)
        if result.acquired:
            return LockHandle(
                key=key,
                owner_id=owner_id,
                fencing_token=result.fencing_token,
                lock_manager=self,
            )
        return None

    def release(
        self,
        key: str,
        owner_id: str,
        fencing_token: int = 0,
    ) -> bool:
        """Release a lock.

        Args:
            key: The lock key.
            owner_id: The owner that acquired the lock.
            fencing_token: The fencing token from acquisition.

        Returns:
            True if the lock was released.
        """
        partition_id = self._strategy.get_partition_id(key)
        owner_address = self._table.get_owner(partition_id)

        if owner_address is None or owner_address == self._self_address:
            return self._lock_store.release(key, owner_id, fencing_token)

        # Forward to remote owner
        try:
            response = self._rpc.lock_release(
                owner_address, key, owner_id, fencing_token
            )
            return response.success and response.payload.get("released", False)
        except Exception:
            logger.exception(
                "Failed to release lock '%s' on %s", key, owner_address
            )
            return False

    def is_locked(self, key: str) -> bool:
        """Check if a key is currently locked."""
        partition_id = self._strategy.get_partition_id(key)
        owner_address = self._table.get_owner(partition_id)

        if owner_address is None or owner_address == self._self_address:
            return self._lock_store.is_locked(key)

        try:
            response = self._rpc.lock_is_locked(owner_address, key)
            return response.success and response.payload.get("is_locked", False)
        except Exception:
            logger.exception(
                "Failed to check lock '%s' on %s", key, owner_address
            )
            return False

    def force_release(self, key: str) -> bool:
        """Force-release a lock regardless of owner."""
        partition_id = self._strategy.get_partition_id(key)
        owner_address = self._table.get_owner(partition_id)

        if owner_address is None or owner_address == self._self_address:
            return self._lock_store.force_release(key)

        try:
            response = self._rpc.lock_force_release(owner_address, key)
            return response.success and response.payload.get("released", False)
        except Exception:
            logger.exception(
                "Failed to force-release lock '%s' on %s", key, owner_address
            )
            return False

    # ── Local handlers (called by RPC server) ─────────────────────

    def handle_lock_acquire(self, payload: dict) -> dict:
        """Handle a remote lock acquire request."""
        result = self._lock_store.try_acquire(
            key=payload["key"],
            owner_id=payload["owner_id"],
            hold_timeout=payload.get("hold_timeout", 60.0),
        )
        return {
            "acquired": result.acquired,
            "owner_id": result.owner_id,
            "fencing_token": result.fencing_token,
            "key": result.key,
        }

    def handle_lock_release(self, payload: dict) -> dict:
        """Handle a remote lock release request."""
        released = self._lock_store.release(
            key=payload["key"],
            owner_id=payload["owner_id"],
            fencing_token=payload.get("fencing_token", 0),
        )
        return {"released": released}

    def handle_lock_is_locked(self, payload: dict) -> dict:
        """Handle a remote is_locked check."""
        return {"is_locked": self._lock_store.is_locked(payload["key"])}

    def handle_lock_force_release(self, payload: dict) -> dict:
        """Handle a remote force-release."""
        return {"released": self._lock_store.force_release(payload["key"])}

    # ── Private ───────────────────────────────────────────────────

    def _try_acquire(
        self, key: str, owner_id: str, hold_timeout: float
    ) -> LockResult:
        """Single attempt to acquire a lock (local or remote)."""
        partition_id = self._strategy.get_partition_id(key)
        owner_address = self._table.get_owner(partition_id)

        if owner_address is None or owner_address == self._self_address:
            return self._lock_store.try_acquire(key, owner_id, hold_timeout)

        # Forward to remote partition owner
        try:
            response = self._rpc.lock_acquire(
                owner_address, key, owner_id, hold_timeout
            )
            payload = response.payload if response.success else {}
            return LockResult(
                acquired=payload.get("acquired", False),
                owner_id=payload.get("owner_id"),
                fencing_token=payload.get("fencing_token", 0),
                key=key,
            )
        except Exception:
            logger.exception(
                "Failed to acquire lock '%s' on %s", key, owner_address
            )
            return LockResult(acquired=False, key=key)