from typing import Optional
from sphana_rag.models import IndexDetails
from .base_db_repository import BaseDbRepository

TABLE_NAME: str = "global"

class IndexDetailsRepository(BaseDbRepository[IndexDetails]):
    def __init__(self):
        db_location: str = "./.database/index_details_db" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        super().__init__(db_location, secondary)
        self.init_table()
        
    def init_table(self) -> None:
        self._init_table(TABLE_NAME)

    def drop_table(self) -> None:
        self._drop_table(TABLE_NAME)

    def upsert(self, index_details: IndexDetails) -> None:
        self._upsert_document(TABLE_NAME, index_details.index_name, index_details)

    def delete(self, index_name: str) -> None:
        self._delete_document(TABLE_NAME, index_name)

    def read(self, index_name: str) -> Optional[IndexDetails]:
        return self._read_document(TABLE_NAME, index_name)
    
    def exists(self, index_name: str) -> bool:
        return self._document_exists(TABLE_NAME, index_name)
    