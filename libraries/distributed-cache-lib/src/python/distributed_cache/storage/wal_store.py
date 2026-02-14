"""Write-Ahead Log (WAL) — append-only disk persistence for durability.

Every cache mutation (put / delete / clear) is written to a WAL file
*before* the response is sent to the caller.  On crash-recovery the
WAL is replayed to restore in-memory state.

WAL files are segmented: when a segment exceeds ``max_segment_bytes``
a new segment is created and old, fully-replayed segments are removed
during compaction.

File format (per record)::

    [4 bytes: record length (big-endian uint32)]
    [4 bytes: CRC32 checksum of the payload (big-endian uint32)]
    [N bytes: msgpack-encoded record dict]

The record dict has the following keys:
    - ``op``: ``"put"`` | ``"delete"`` | ``"clear"``
    - ``pid``: partition_id (int)
    - ``col``: collection name (str)
    - ``key``: cache key (str) — absent for ``clear``
    - ``val``: cache value (Any) — only for ``put``
    - ``ttl``: TTL seconds (float | None) — only for ``put``
    - ``ts``:  epoch timestamp of the operation (float)
    - ``seq``: monotonically increasing sequence number (int)
"""

from __future__ import annotations

import logging
import os
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable

import msgpack

logger = logging.getLogger(__name__)

_HEADER_FMT = ">II"  # big-endian uint32 length + uint32 crc32
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

class WalStore:
    """Append-only WAL for crash-recovery of cache data.

    Parameters:
        wal_dir: Directory for WAL segment files.
        max_segment_bytes: Max size of a single WAL segment file
            before rotation (default 64 MiB).
        fsync: Whether to ``os.fsync`` after every write for
            guaranteed durability. Setting to False trades durability
            for throughput.
        enabled: Set to False to disable WAL entirely (pure in-memory).
    """

    def __init__(
        self,
        wal_dir: str | Path,
        max_segment_bytes: int = 64 * 1024 * 1024,
        fsync: bool = True,
        enabled: bool = True,
    ) -> None:
        self._wal_dir = Path(wal_dir)
        self._max_segment_bytes = max_segment_bytes
        self._fsync = fsync
        self._enabled = enabled

        self._lock = threading.Lock()
        self._seq: int = 0
        self._current_segment: _WalSegment | None = None
        self._replayed_up_to_seq: int = 0

        if self._enabled:
            self._wal_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def sequence(self) -> int:
        return self._seq

    # ── Write operations ──────────────────────────────────────────

    def log_put(
        self,
        partition_id: int,
        collection: str,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> int:
        """Append a put record to the WAL.

        Returns:
            The sequence number of this record.
        """
        if not self._enabled:
            return 0
        return self._append({
            "op": "put",
            "pid": partition_id,
            "col": collection,
            "key": key,
            "val": value,
            "ttl": ttl,
        })

    def log_delete(
        self,
        partition_id: int,
        collection: str,
        key: str,
    ) -> int:
        """Append a delete record to the WAL."""
        if not self._enabled:
            return 0
        return self._append({
            "op": "delete",
            "pid": partition_id,
            "col": collection,
            "key": key,
        })

    def log_clear(
        self,
        partition_id: int,
        collection: str,
    ) -> int:
        """Append a clear record to the WAL."""
        if not self._enabled:
            return 0
        return self._append({
            "op": "clear",
            "pid": partition_id,
            "col": collection,
        })

    # ── Replay ────────────────────────────────────────────────────

    def replay(
        self,
        on_put: Callable[[int, str, str, Any, float | None], None],
        on_delete: Callable[[int, str, str], None],
        on_clear: Callable[[int, str], None],
    ) -> int:
        """Replay all WAL records in order, invoking the appropriate
        callback for each operation.

        Args:
            on_put: ``(partition_id, collection, key, value, ttl)``
            on_delete: ``(partition_id, collection, key)``
            on_clear: ``(partition_id, collection)``

        Returns:
            Number of records replayed.
        """
        if not self._enabled:
            return 0

        segments = self._list_segments()
        if not segments:
            logger.info("WAL replay: no segments found")
            return 0

        total = 0
        max_seq = 0

        for seg_path in segments:
            try:
                for record in _read_segment(seg_path):
                    seq = record.get("seq", 0)
                    op = record.get("op")
                    pid = record.get("pid", 0)
                    col = record.get("col", "__default__")

                    if op == "put":
                        on_put(
                            pid,
                            col,
                            record["key"],
                            record.get("val"),
                            record.get("ttl"),
                        )
                    elif op == "delete":
                        on_delete(pid, col, record["key"])
                    elif op == "clear":
                        on_clear(pid, col)
                    else:
                        logger.warning("Unknown WAL op: %s", op)
                        continue

                    total += 1
                    if seq > max_seq:
                        max_seq = seq

            except Exception:
                logger.exception(
                    "Error replaying WAL segment %s", seg_path
                )

        with self._lock:
            self._seq = max_seq
            self._replayed_up_to_seq = max_seq

        logger.info(
            "WAL replay complete: %d records, max_seq=%d", total, max_seq
        )
        return total

    # ── Compaction ────────────────────────────────────────────────

    def compact(self) -> int:
        """Remove fully-replayed WAL segments.

        Keeps only the current (active) segment.

        Returns:
            Number of segments removed.
        """
        if not self._enabled:
            return 0

        removed = 0
        segments = self._list_segments()

        with self._lock:
            current_name = (
                self._current_segment.path.name
                if self._current_segment
                else None
            )

        for seg_path in segments:
            if seg_path.name == current_name:
                continue
            try:
                seg_path.unlink()
                removed += 1
            except OSError:
                logger.warning("Failed to remove WAL segment %s", seg_path)

        if removed:
            logger.info("WAL compaction: removed %d old segments", removed)
        return removed

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the current WAL segment."""
        with self._lock:
            if self._current_segment is not None:
                self._current_segment.close()
                self._current_segment = None

    # ── Private ───────────────────────────────────────────────────

    def _append(self, record: dict) -> int:
        with self._lock:
            self._seq += 1
            record["seq"] = self._seq
            record["ts"] = time.time()

            seg = self._ensure_segment()
            seg.write(record)
            return self._seq

    def _ensure_segment(self) -> _WalSegment:
        """Return the active segment, rotating if needed.

        Must be called while holding ``self._lock``.
        """
        if (
            self._current_segment is None
            or self._current_segment.size >= self._max_segment_bytes
        ):
            if self._current_segment is not None:
                self._current_segment.close()
            name = f"wal_{int(time.time() * 1000):016d}.seg"
            path = self._wal_dir / name
            self._current_segment = _WalSegment(path, self._fsync)
        return self._current_segment

    def _list_segments(self) -> list[Path]:
        """Return sorted list of WAL segment files."""
        if not self._wal_dir.exists():
            return []
        return sorted(self._wal_dir.glob("wal_*.seg"))


# ── WAL segment helper ───────────────────────────────────────────

class _WalSegment:
    """A single WAL segment file."""

    def __init__(self, path: Path, fsync: bool = True) -> None:
        self.path = path
        self._fsync = fsync
        self._fd = open(path, "ab")  # noqa: SIM115
        self._size = path.stat().st_size if path.exists() else 0

    @property
    def size(self) -> int:
        return self._size

    def write(self, record: dict) -> None:
        data = msgpack.packb(record, use_bin_type=True)
        crc = zlib.crc32(data) & 0xFFFFFFFF
        header = struct.pack(_HEADER_FMT, len(data), crc)
        self._fd.write(header + data)
        if self._fsync:
            self._fd.flush()
            os.fsync(self._fd.fileno())
        self._size += _HEADER_SIZE + len(data)

    def close(self) -> None:
        try:
            self._fd.flush()
            if self._fsync:
                os.fsync(self._fd.fileno())
            self._fd.close()
        except OSError:
            pass


def _read_segment(path: Path) -> list[dict]:
    """Read all records from a WAL segment file.

    Each record is validated with a CRC32 checksum.  Records that
    fail validation are skipped with a warning.
    """
    records: list[dict] = []
    with open(path, "rb") as f:
        while True:
            header = f.read(_HEADER_SIZE)
            if len(header) < _HEADER_SIZE:
                break
            length, expected_crc = struct.unpack(_HEADER_FMT, header)
            data = f.read(length)
            if len(data) < length:
                logger.warning(
                    "Truncated WAL record in %s (expected %d, got %d)",
                    path,
                    length,
                    len(data),
                )
                break

            # Validate CRC32
            actual_crc = zlib.crc32(data) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                logger.warning(
                    "CRC mismatch in WAL segment %s at offset %d "
                    "(expected 0x%08x, got 0x%08x) — skipping record",
                    path,
                    f.tell() - length - _HEADER_SIZE,
                    expected_crc,
                    actual_crc,
                )
                continue

            try:
                record = msgpack.unpackb(data, raw=False)
                records.append(record)
            except Exception:
                logger.warning("Corrupt WAL record in %s", path)
                break
    return records
