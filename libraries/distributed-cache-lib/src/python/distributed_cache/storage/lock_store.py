"""Local in-memory lock store.

Manages distributed lock state for partitions owned by this node.
Each lock is keyed and has an owner, fencing token, and hold timeout.
Expired locks are automatically released on access.
"""

from __future__ import annotations

import logging
import threading
import time

from ..models import LockEntry, LockResult

logger = logging.getLogger(__name__)

class LockStore:
    """Thread-safe in-memory store for distributed locks.

    Locks are partitioned the same way cache keys are — the lock key
    is hashed to a partition, and only the partition owner manages
    the lock state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._locks: dict[str, LockEntry] = {}
        self._fencing_counter = 0

    def _next_fencing_token(self) -> int:
        self._fencing_counter += 1
        return self._fencing_counter

    def try_acquire(
        self,
        key: str,
        owner_id: str,
        hold_timeout: float = 60.0,
    ) -> LockResult:
        """Attempt to acquire a lock.

        If the lock is free or expired, it is granted to the caller.
        If already held by the same owner, it is re-entrant (refreshed).

        Args:
            key: The lock key.
            owner_id: Identifier of the requester.
            hold_timeout: Max seconds the lock may be held.

        Returns:
            A :class:`LockResult` indicating success/failure.
        """
        with self._lock:
            existing = self._locks.get(key)

            # Clean up expired lock
            if existing is not None and existing.is_expired:
                logger.debug(
                    "Lock '%s' expired (owner=%s), releasing",
                    key,
                    existing.owner_id,
                )
                del self._locks[key]
                existing = None

            # Lock is free
            if existing is None:
                token = self._next_fencing_token()
                self._locks[key] = LockEntry(
                    key=key,
                    owner_id=owner_id,
                    fencing_token=token,
                    acquired_at=time.time(),
                    hold_timeout=hold_timeout,
                )
                return LockResult(
                    acquired=True,
                    owner_id=owner_id,
                    fencing_token=token,
                    key=key,
                )

            # Re-entrant: same owner refreshes the lock
            if existing.owner_id == owner_id:
                existing.acquired_at = time.time()
                existing.hold_timeout = hold_timeout
                return LockResult(
                    acquired=True,
                    owner_id=owner_id,
                    fencing_token=existing.fencing_token,
                    key=key,
                )

            # Lock held by someone else
            return LockResult(
                acquired=False,
                owner_id=existing.owner_id,
                fencing_token=existing.fencing_token,
                key=key,
            )

    def release(
        self,
        key: str,
        owner_id: str,
        fencing_token: int = 0,
    ) -> bool:
        """Release a lock.

        The lock is only released if the owner matches.
        Optionally validates the fencing token (if > 0).

        Returns:
            True if the lock was released, False otherwise.
        """
        with self._lock:
            existing = self._locks.get(key)
            if existing is None:
                return True  # Already free

            if existing.is_expired:
                del self._locks[key]
                return True

            if existing.owner_id != owner_id:
                logger.warning(
                    "Lock '%s' release denied: owner mismatch "
                    "(requested=%s, actual=%s)",
                    key,
                    owner_id,
                    existing.owner_id,
                )
                return False

            if fencing_token > 0 and existing.fencing_token != fencing_token:
                logger.warning(
                    "Lock '%s' release denied: fencing token mismatch "
                    "(requested=%d, actual=%d)",
                    key,
                    fencing_token,
                    existing.fencing_token,
                )
                return False

            del self._locks[key]
            return True

    def force_release(self, key: str) -> bool:
        """Force-release a lock regardless of owner."""
        with self._lock:
            if key in self._locks:
                del self._locks[key]
                return True
            return False

    def is_locked(self, key: str) -> bool:
        """Check if a key is currently locked (and not expired)."""
        with self._lock:
            entry = self._locks.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._locks[key]
                return False
            return True

    def get_owner(self, key: str) -> str | None:
        """Return the owner of a lock, or None if not locked."""
        with self._lock:
            entry = self._locks.get(key)
            if entry is None or entry.is_expired:
                return None
            return entry.owner_id

    def evict_expired(self) -> int:
        """Remove all expired locks. Returns count removed."""
        evicted = 0
        with self._lock:
            expired = [k for k, v in self._locks.items() if v.is_expired]
            for k in expired:
                del self._locks[k]
                evicted += 1
        return evicted

    def active_lock_count(self) -> int:
        """Return the number of active (non-expired) locks."""
        with self._lock:
            return sum(1 for v in self._locks.values() if not v.is_expired)

    def all_locks(self) -> dict[str, LockEntry]:
        """Return a snapshot of all active locks."""
        with self._lock:
            return {
                k: v for k, v in self._locks.items() if not v.is_expired
            }