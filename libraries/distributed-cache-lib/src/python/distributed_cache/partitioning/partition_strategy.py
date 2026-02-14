"""Consistent-hashing partition strategy.

Maps cache keys to partition IDs using a stable hash function.
The number of partitions is fixed at cluster creation and does NOT
change when nodes join/leave — only partition *ownership* changes.
"""

from __future__ import annotations

import hashlib

# Default number of partitions (similar to Hazelcast's 271)
DEFAULT_PARTITION_COUNT = 271

class PartitionStrategy:
    """Maps keys to fixed partition IDs via consistent hashing.

    Parameters:
        partition_count: Total number of partitions.  Must remain
            constant for the lifetime of the cache cluster.
    """

    def __init__(self, partition_count: int = DEFAULT_PARTITION_COUNT) -> None:
        if partition_count < 1:
            raise ValueError("partition_count must be >= 1")
        self._partition_count = partition_count

    @property
    def partition_count(self) -> int:
        return self._partition_count

    def get_partition_id(self, key: str) -> int:
        """Return the partition ID for the given cache key.

        Uses MD5 for fast, uniform distribution (not cryptographic).
        """
        digest = hashlib.md5(key.encode("utf-8")).digest()
        # Use first 4 bytes as unsigned int
        hash_val = int.from_bytes(digest[:4], byteorder="big", signed=False)
        return hash_val % self._partition_count

    def get_partition_ids_for_keys(self, keys: list[str]) -> dict[int, list[str]]:
        """Group a list of keys by their partition ID.

        Returns:
            Dict mapping partition_id → list of keys belonging to that
            partition.
        """
        result: dict[int, list[str]] = {}
        for key in keys:
            pid = self.get_partition_id(key)
            result.setdefault(pid, []).append(key)
        return result