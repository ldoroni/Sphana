import threading
from typing import Optional
from injector import singleton
from rocksdict import Rdict
from sphana_rag.models import ParentChunkDetails
from sphana_rag.models import ListResults
from .base_documents_repository import BaseDocumentsRepository

@singleton
class ParentChunkDetailsRepository(BaseDocumentsRepository[ParentChunkDetails]):
    
    def __init__(self):
        self.__last_unique_id_map: dict[str, int] = {}
        self.__last_unique_id_lock = threading.Lock()
        db_location: str = "./.database/parent_chunk_details" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)

    def init_table(self, shard_name: str) -> None:
        self._init_table(shard_name)

    def drop_table(self, shard_name: str) -> None:
        self._drop_table(shard_name)

    def upsert(self, shard_name: str, parent_chunk_details: ParentChunkDetails) -> None:
        self._upsert_document(shard_name, parent_chunk_details.parent_chunk_id, parent_chunk_details)

    def delete(self, shard_name: str, parent_chunk_id: str) -> None:
        self._delete_document(shard_name, parent_chunk_id)

    def read(self, shard_name: str, parent_chunk_id: str) -> Optional[ParentChunkDetails]:
        return self._read_document(shard_name, parent_chunk_id)
    
    def list(self, shard_name: str, offset: Optional[str], limit: int) -> ListResults[ParentChunkDetails]:
        return self._list_documents(shard_name, offset, limit)
    
    def exists(self, shard_name: str, parent_chunk_id: str) -> bool:
        return self._document_exists(shard_name, parent_chunk_id)
    
    def next_unique_id(self, shard_name: str) -> str:
        with self.__last_unique_id_lock:
            # Read last unique ID from cache
            cached_latest_id: Optional[int] = self.__last_unique_id_map.get(shard_name)
            if cached_latest_id is not None:
                next_id = cached_latest_id + 1
                self.__last_unique_id_map[shard_name] = next_id
                return str(next_id)
            
            try:
                # Read last unique ID from DB
                # items(backwards=True) starts at the highest key
                # We wrap it in next() to get just the first (highest) element
                table: Rdict = super()._get_table(shard_name)
                latest_id = next(table.keys(backwards=True))
                next_id = int(latest_id) + 1
                self.__last_unique_id_map[shard_name] = next_id
                return str(next_id)
            except StopIteration:
                # The database is empty
                # Initialize the unique ID counter 
                next_id = 0
                self.__last_unique_id_map[shard_name] = next_id
                return str(next_id)