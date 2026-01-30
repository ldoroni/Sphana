from typing import Optional
from injector import singleton
from sphana_rag.models import DocumentDetails
from .base_db_repository import BaseDbRepository

@singleton
class DocumentDetailsRepository(BaseDbRepository[DocumentDetails]):
    
    def __init__(self):
        db_location: str = "./.database/document_details_db" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)

    def init_table(self, index_name: str) -> None:
        self._init_table(index_name)

    def drop_table(self, index_name: str) -> None:
        self._drop_table(index_name)

    def upsert(self, index_name: str, document_details: DocumentDetails) -> None:
        self._upsert_document(index_name, document_details.document_id, document_details)

    def delete(self, index_name: str, document_id: str) -> None:
        self._delete_document(index_name, document_id)

    def read(self, index_name: str, document_id: str) -> Optional[DocumentDetails]:
        return self._read_document(index_name, document_id)
    
    def exists(self, index_name: str, document_id: str) -> bool:
        return self._document_exists(index_name, document_id)
    