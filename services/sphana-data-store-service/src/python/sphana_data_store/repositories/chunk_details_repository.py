from typing import Optional
from injector import singleton
from rocksdict import Rdict
from sphana_data_store.models import ChunkDetails
from sphana_data_store.models import ListResults
from .base_documents_repository import BaseDocumentsRepository

@singleton
class ChunkDetailsRepository(BaseDocumentsRepository[ChunkDetails]):
    
    def __init__(self):
        db_location: str = "./.database/chunk_details" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)

    def init_table(self, shard_name: str) -> None:
        self._init_table(shard_name)

    def drop_table(self, shard_name: str) -> None:
        self._drop_table(shard_name)
    
    def upsert(self, shard_name: str, chunk_details: ChunkDetails) -> None:
        self._upsert_document(shard_name, chunk_details.chunk_id, chunk_details)

    def delete(self, shard_name: str, chunk_id: str) -> None:
        self._delete_document(shard_name, chunk_id)

    def read(self, shard_name: str, chunk_id: str) -> Optional[ChunkDetails]:
        return self._read_document(shard_name, chunk_id)
    
    def list_for_entry(self, shard_name: str, entry_id: str) -> list[ChunkDetails]:
        table: Rdict = self._get_table(shard_name)
        items = table.items(from_key=f"{entry_id}:")
        documents: list[ChunkDetails] = []
        for key, value in items:
            if not str(key).startswith(f"{entry_id}:"):
                break
            elif value.entry_id == entry_id:
                documents.append(value)
        return documents
    
    def list(self, shard_name: str, offset: Optional[str], limit: int) -> ListResults[ChunkDetails]:
        return self._list_documents(shard_name, offset, limit)

    def exists(self, shard_name: str, chunk_id: str) -> bool:
        return self._document_exists(shard_name, chunk_id)
    
    def new_chunk_id(self, shard_name: str, entry_id: str) -> str:
        table: Rdict = self._get_table(shard_name)
        ceiling_key = f"{entry_id};" # TODO: ensure entity_id does not contain ':' or ';' characters!
        items = table.items(from_key=ceiling_key, backwards=True)    
        try:
            last_key, _ = next(items)
            if str(last_key).startswith(f"{entry_id}:"):
                # The last key is in the format <entry_id>:<chunk_id>, so we can extract the chunk_id and increment it
                last_num = int(str(last_key).split(":")[-1])
                next_num = last_num + 1
            else:
                # No chunks exist for this entry_id yet
                next_num = 0
        except StopIteration:
            next_num = 0
        return f"{entry_id}:{next_num:09d}" # TODO: ensure max number of chunks per entry does not exceed 1,000,000,000 (i.e. 10^9)
    