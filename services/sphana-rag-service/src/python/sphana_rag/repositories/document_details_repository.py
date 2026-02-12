from typing import Optional
from injector import singleton
from sphana_rag.models import DocumentDetails
from sphana_rag.models import ListResults
from .base_documents_repository import BaseDocumentsRepository

@singleton
class DocumentDetailsRepository(BaseDocumentsRepository[DocumentDetails]):
    
    def __init__(self):
        db_location: str = "./.database/document_details" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)

    def init_table(self, shard_name: str) -> None:
        self._init_table(shard_name)

    def drop_table(self, shard_name: str) -> None:
        self._drop_table(shard_name)

    def upsert(self, shard_name: str, document_details: DocumentDetails) -> None:
        self._upsert_document(shard_name, document_details.document_id, document_details)

    def delete(self, shard_name: str, document_id: str) -> None:
        self._delete_document(shard_name, document_id)

    def read(self, shard_name: str, document_id: str) -> Optional[DocumentDetails]:
        return self._read_document(shard_name, document_id)
    
    def list(self, shard_name: str, offset: Optional[str], limit: int) -> ListResults[DocumentDetails]:
        return self._list_documents(shard_name, offset, limit)
    
    def exists(self, shard_name: str, document_id: str) -> bool:
        return self._document_exists(shard_name, document_id)
    