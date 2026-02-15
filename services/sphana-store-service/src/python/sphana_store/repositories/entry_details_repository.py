from typing import Optional
from injector import singleton
from sphana_store.models import EntryDetails, ListResults
from .base_documents_repository import BaseDocumentsRepository

@singleton
class EntryDetailsRepository(BaseDocumentsRepository[EntryDetails]):
    
    def __init__(self):
        db_location: str = "./.database/entry_details" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)

    def init_table(self, shard_name: str) -> None:
        self._init_table(shard_name)

    def drop_table(self, shard_name: str) -> None:
        self._drop_table(shard_name)

    def upsert(self, shard_name: str, entry_details: EntryDetails) -> None:
        self._upsert_document(shard_name, entry_details.entry_id, entry_details)

    def delete(self, shard_name: str, entry_id: str) -> None:
        self._delete_document(shard_name, entry_id)

    def read(self, shard_name: str, entry_id: str) -> Optional[EntryDetails]:
        return self._read_document(shard_name, entry_id)
    
    def list(self, shard_name: str, offset: Optional[str], limit: int) -> ListResults[EntryDetails]:
        return self._list_documents(shard_name, offset, limit)
    
    def exists(self, shard_name: str, entry_id: str) -> bool:
        return self._document_exists(shard_name, entry_id)
    