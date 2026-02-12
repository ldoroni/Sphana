from typing import Optional
from injector import singleton
from sphana_rag.constants import GLOBAL_SHARD_NAME
from sphana_rag.models import ShardDetails
from sphana_rag.models import ListResults
from .base_documents_repository import BaseDocumentsRepository

@singleton
class ShardDetailsRepository(BaseDocumentsRepository[ShardDetails]):
    
    def __init__(self):
        db_location: str = "./.database/shard_details" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)
        self._init_table(GLOBAL_SHARD_NAME)
        
    # def init_table(self) -> None:
    #     self._init_table(GLOBAL_SHARD_NAME)

    # def drop_table(self) -> None:
    #     self._drop_table(GLOBAL_SHARD_NAME)

    def upsert(self, shard_details: ShardDetails) -> None:
        self._upsert_document(GLOBAL_SHARD_NAME, shard_details.shard_name, shard_details)

    def delete(self, shard_name: str) -> None:
        self._delete_document(GLOBAL_SHARD_NAME, shard_name)

    def read(self, shard_name: str) -> Optional[ShardDetails]:
        return self._read_document(GLOBAL_SHARD_NAME, shard_name)
    
    def list(self, offset: Optional[str], limit: int) -> ListResults[ShardDetails]:
        return self._list_documents(GLOBAL_SHARD_NAME, offset, limit)
    
    def exists(self, shard_name: str) -> bool:
        return self._document_exists(GLOBAL_SHARD_NAME, shard_name)
    