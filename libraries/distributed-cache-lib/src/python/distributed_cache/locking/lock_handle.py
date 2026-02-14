"""Lock handle — RAII-style context manager for distributed locks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lock_manager import LockManager

logger = logging.getLogger(__name__)


class LockHandle:
    """A handle to a held distributed lock.

    Supports ``with`` statement for automatic release::

        handle = cache.acquire_lock("my-key")
        with handle:
            # critical section
        # lock released automatically

    Parameters:
        key: The lock key.
        owner_id: Identifier of the lock holder.
        fencing_token: Monotonically increasing token.
        lock_manager: The :class:`LockManager` that owns this lock.
    """

    def __init__(
        self,
        key: str,
        owner_id: str,
        fencing_token: int,
        lock_manager: LockManager,
    ) -> None:
        self._key = key
        self._owner_id = owner_id
        self._fencing_token = fencing_token
        self._manager = lock_manager
        self._released = False

    @property
    def key(self) -> str:
        return self._key

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def fencing_token(self) -> int:
        return self._fencing_token

    @property
    def is_released(self) -> bool:
        return self._released

    def release(self) -> bool:
        """Release this lock.

        Returns:
            True if the lock was successfully released.
        """
        if self._released:
            return True
        try:
            result = self._manager.release(
                self._key, self._owner_id, self._fencing_token
            )
            self._released = result
            return result
        except Exception:
            logger.exception("Failed to release lock '%s'", self._key)
            return False

    def __enter__(self) -> LockHandle:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.release()

    def __repr__(self) -> str:
        state = "released" if self._released else "held"
        return (
            f"LockHandle(key={self._key!r}, owner={self._owner_id!r}, "
            f"token={self._fencing_token}, {state})"
        )