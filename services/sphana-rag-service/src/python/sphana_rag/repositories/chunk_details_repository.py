from typing import Optional
from sphana_rag.models import ChunkDetails
from .base_db_repository import BaseDbRepository

class ChunkDetailsRepository(BaseDbRepository[ChunkDetails]):
    def __init__(self):
        db_location: str = "./.database/chunk_details_db" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)

    def init_table(self, index_name: str) -> None:
        self._init_table(index_name)

    def drop_table(self, index_name: str) -> None:
        self._drop_table(index_name)

    def upsert(self, index_name: str, chunk_details: ChunkDetails) -> None:
        self._upsert_document(index_name, chunk_details.chunk_id, chunk_details)

    def delete(self, index_name: str, chunk_id: str) -> None:
        self._delete_document(index_name, chunk_id)

    def read(self, index_name: str, chunk_id: str) -> Optional[ChunkDetails]:
        return self._read_document(index_name, chunk_id)
    
    def exists(self, index_name: str, chunk_id: str) -> bool:
        return self._document_exists(index_name, chunk_id)
    