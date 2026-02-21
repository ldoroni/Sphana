import logging
import shutil
from abc import ABC
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import zarr
from prometheus_client import Counter, Histogram

from sphana_data_store.models import ListResults
from sphana_data_store.utils import Base64Util

logger = logging.getLogger(__name__)

ZARR_EXE_COUNTER = Counter(
    "spn_zarr_exe_total",
    "Total number of Zarr operations executed",
    ["storage", "operation"],
)
ZARR_EXE_DURATION_HISTOGRAM = Histogram(
    "spn_zarr_exe_duration_seconds",
    "Duration of Zarr operations in seconds",
    ["storage", "operation"],
)

# Zarr array configuration for blob storage
_CHUNK_SIZE: int = 1024 * 1024            # 1 MB logical chunks
_SHARD_SIZE: int = 100 * 1024 * 1024      # 100 MB shards (packs 100 chunks per file on PVC)
_DTYPE = "u1"                              # Unsigned 8-bit integers (raw bytes)


class BaseBlobRepository(ABC):
    """Abstract base repository for binary blob storage backed by Zarr groups.

    Each *storage* is a Zarr group on disk.  Individual blobs are stored as
    separate 1-D ``uint8`` arrays inside the group, keyed by ``blob_id``.

    For large blobs (GiBs) use the chunked API:
    - Write: call ``_write_blob_chunk`` repeatedly to stream data in.
    - Read:  call ``_read_blob_chunk`` with byte ranges to stream data out.
    - Size:  call ``_get_blob_size`` to obtain the total length without reading data.
    """

    def __init__(self, db_location: str, secondary: bool) -> None:
        self.__db_location: str = db_location
        self.__storage_map: dict[str, zarr.Group] = {}
        self.__secondary: bool = secondary  # Reserved for future use

    # ------------------------------------------------------------------
    # Storage lifecycle
    # ------------------------------------------------------------------

    def _init_storage(self, storage_name: str) -> None:
        """Ensure the storage group exists (creates it if missing)."""
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="init_storage").inc()
        try:
            self._get_storage(storage_name)
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="init_storage").observe(duration)

    def _drop_storage(self, storage_name: str) -> None:
        """Delete the storage group and all its blobs from disk."""
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="drop_storage").inc()
        try:
            storage_location: str = self.__get_storage_location(storage_name)
            storage_path = Path(storage_location)
            shutil.rmtree(storage_path, ignore_errors=True)
            self.__storage_map.pop(storage_name, None)
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="drop_storage").observe(duration)

    # ------------------------------------------------------------------
    # Blob write
    # ------------------------------------------------------------------

    def _write_blob(self, storage_name: str, blob_id: str, buffer: bytes) -> None:
        """Create or overwrite a blob in the storage group.

        The entire *buffer* must fit in memory.  For large blobs prefer
        ``_write_blob_chunk`` which appends incrementally.
        """
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="write_blob").inc()
        try:
            group: zarr.Group = self._get_storage(storage_name)
            data: np.ndarray = np.frombuffer(buffer, dtype=_DTYPE)
            group.create_array(
                name=blob_id,
                data=data,
                chunks=(_CHUNK_SIZE,),
                shards=(_SHARD_SIZE,),
                overwrite=True,
            )
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="write_blob").observe(duration)

    def _write_blob_chunk(self, storage_name: str, blob_id: str, buffer: bytes) -> None:
        """Append a chunk of bytes to a blob.  Creates the blob if it does not exist.

        Call this method repeatedly to stream large buffers into storage
        without holding the full blob in memory.
        """
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="write_blob_chunk").inc()
        try:
            group: zarr.Group = self._get_storage(storage_name)
            new_data: np.ndarray = np.frombuffer(buffer, dtype=_DTYPE)

            if blob_id in group:
                existing: zarr.Array = group[blob_id]  # type: ignore[assignment]
                old_size: int = existing.shape[0]
                new_size: int = old_size + new_data.shape[0]
                existing.resize(new_size)
                existing[old_size:new_size] = new_data
            else:
                group.create_array(
                    name=blob_id,
                    data=new_data,
                    chunks=(_CHUNK_SIZE,),
                    shards=(_SHARD_SIZE,),
                )
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="write_blob_chunk").observe(duration)

    # ------------------------------------------------------------------
    # Blob read
    # ------------------------------------------------------------------

    def _read_blob(self, storage_name: str, blob_id: str) -> Optional[bytes]:
        """Read the full contents of a blob, or ``None`` if it does not exist.

        The entire blob is materialised in memory.  For large blobs prefer
        ``_read_blob_chunk`` which reads a byte range at a time.
        """
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="read_blob").inc()
        try:
            group: zarr.Group = self._get_storage(storage_name)
            if blob_id not in group:
                return None
            arr: zarr.Array = group[blob_id]  # type: ignore[assignment]
            return np.asarray(arr[:]).tobytes()
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="read_blob").observe(duration)

    def _read_blob_chunk(
        self,
        storage_name: str,
        blob_id: str,
        start_index: int,
        end_index: int,
    ) -> Optional[bytes]:
        """Read a byte-range ``[start_index:end_index]`` of a blob.

        Only the requested range is loaded into memory, making this suitable
        for streaming reads of large blobs.  Returns ``None`` if the blob
        does not exist.
        """
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="read_blob_chunk").inc()
        try:
            group: zarr.Group = self._get_storage(storage_name)
            if blob_id not in group:
                return None
            arr: zarr.Array = group[blob_id]  # type: ignore[assignment]
            return np.asarray(arr[start_index:end_index]).tobytes()
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="read_blob_chunk").observe(duration)

    # ------------------------------------------------------------------
    # Blob delete
    # ------------------------------------------------------------------

    def _delete_blob(self, storage_name: str, blob_id: str) -> None:
        """Delete a blob from the storage group."""
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="delete_blob").inc()
        try:
            group: zarr.Group = self._get_storage(storage_name)
            if blob_id in group:
                del group[blob_id]
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="delete_blob").observe(duration)

    # ------------------------------------------------------------------
    # Metadata, listing & existence
    # ------------------------------------------------------------------

    def _get_blob_size(self, storage_name: str, blob_id: str) -> Optional[int]:
        """Return the size in bytes of a blob, or ``None`` if it does not exist.

        Reads only Zarr array metadata — no buffer data is loaded.
        """
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="get_blob_size").inc()
        try:
            group: zarr.Group = self._get_storage(storage_name)
            if blob_id not in group:
                return None
            arr: zarr.Array = group[blob_id]  # type: ignore[assignment]
            return arr.shape[0]
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="get_blob_size").observe(duration)

    def _list_blobs(
        self,
        storage_name: str,
        offset: Optional[str],
        limit: int,
    ) -> ListResults[str]:
        """Return a paginated list of blob IDs stored in the group."""
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="list_blobs").inc()
        try:
            group: zarr.Group = self._get_storage(storage_name)
            all_keys: list[str] = sorted(group.keys())

            # Determine the starting position based on the decoded offset
            plain_offset: Optional[str] = Base64Util.decode_nullable_to_str(offset)
            if plain_offset is not None:
                try:
                    start_idx: int = all_keys.index(plain_offset)
                except ValueError:
                    # Offset key was deleted; find the next key lexicographically
                    start_idx = 0
                    for i, key in enumerate(all_keys):
                        if key >= plain_offset:
                            start_idx = i
                            break
                    else:
                        # All keys are before the offset → empty result
                        return ListResults[str](documents=[], next_offset=None, completed=True)
            else:
                start_idx = 0

            page: list[str] = all_keys[start_idx : start_idx + limit]
            has_more: bool = (start_idx + limit) < len(all_keys)

            next_offset: Optional[str] = None
            if has_more:
                next_offset = Base64Util.encode_nullable_to_str(all_keys[start_idx + limit])

            return ListResults[str](
                documents=page,
                next_offset=next_offset,
                completed=not has_more,
            )
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="list_blobs").observe(duration)

    def _blob_exists(self, storage_name: str, blob_id: str) -> bool:
        """Check whether a blob exists in the storage group."""
        start_time: float = time()
        ZARR_EXE_COUNTER.labels(storage=storage_name, operation="blob_exists").inc()
        try:
            group: zarr.Group = self._get_storage(storage_name)
            return blob_id in group
        finally:
            duration: float = time() - start_time
            ZARR_EXE_DURATION_HISTOGRAM.labels(storage=storage_name, operation="blob_exists").observe(duration)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_storage(self, storage_name: str) -> zarr.Group:
        """Open or create the Zarr group for *storage_name* (cached)."""
        if storage_name in self.__storage_map:
            return self.__storage_map[storage_name]

        # TODO: need lock
        storage_location: str = self.__get_storage_location(storage_name)
        group: zarr.Group = zarr.open_group(store=storage_location, mode="a")
        self.__storage_map[storage_name] = group
        return group

    def __get_storage_location(self, storage_name: str) -> str:
        return f"{self.__db_location}/{storage_name}/blobs.zarr"