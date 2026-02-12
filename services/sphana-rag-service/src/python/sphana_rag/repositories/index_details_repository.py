from typing import Optional
from injector import singleton
from sphana_rag.models import IndexDetails
from sphana_rag.models import ListResults
from .base_documents_repository import BaseDocumentsRepository

SHARD_NAME: str = "global"

@singleton
class IndexDetailsRepository(BaseDocumentsRepository[IndexDetails]):
    
    def __init__(self):
        db_location: str = "./.database/index_details" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)
        self._init_table(SHARD_NAME)
        
    # def init_table(self) -> None:
    #     self._init_table(SHARD_NAME)

    # def drop_table(self) -> None:
    #     self._drop_table(SHARD_NAME)

    def upsert(self, index_details: IndexDetails) -> None:
        self._upsert_document(SHARD_NAME, index_details.index_name, index_details)

    def delete(self, index_name: str) -> None:
        self._delete_document(SHARD_NAME, index_name)

    def read(self, index_name: str) -> Optional[IndexDetails]:
        return self._read_document(SHARD_NAME, index_name)
    
    def list(self, offset: Optional[str], limit: int) -> ListResults[IndexDetails]:
        return self._list_documents(SHARD_NAME, offset, limit)
    
    def exists(self, index_name: str) -> bool:
        return self._document_exists(SHARD_NAME, index_name)
    