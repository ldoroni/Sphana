from typing import Optional
from injector import singleton
from sphana_data_store.models import ListResults
from .base_blob_repository import BaseBlobRepository

@singleton
class EntryPayloadsRepository(BaseBlobRepository):
    
    def __init__(self):
        db_location: str = "./.database/entry_payloads" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)

    def init_storage(self, shard_name: str) -> None:
        self._init_storage(shard_name)

    def drop_storage(self, shard_name: str) -> None:
        self._drop_storage(shard_name)

    def save(self, shard_name: str, entry_id: str, payload: bytes) -> None:
        self._write_blob(shard_name, entry_id, payload)

    def append(self, shard_name: str, entry_id: str, payload: bytes) -> None:
        self._write_blob_chunk(shard_name, entry_id, payload)

    def delete(self, shard_name: str, entry_id: str) -> None:
        self._delete_blob(shard_name, entry_id)

    def read(self, shard_name: str, entry_id: str) -> Optional[bytes]:
        return self._read_blob(shard_name, entry_id)
    
    def read_chunk(self, shard_name: str, entry_id: str, start_index: int, end_index: int) -> Optional[bytes]:
        return self._read_blob_chunk(shard_name, entry_id, start_index, end_index)

    def get_size(self, shard_name: str, entry_id: str) -> Optional[int]:
        return self._get_blob_size(shard_name, entry_id)
    
    def list(self, shard_name: str, offset: Optional[str], limit: int) -> ListResults[str]:
        return self._list_blobs(shard_name, offset, limit)
    
    def exists(self, shard_name: str, entry_id: str) -> bool:
        return self._blob_exists(shard_name, entry_id)
    