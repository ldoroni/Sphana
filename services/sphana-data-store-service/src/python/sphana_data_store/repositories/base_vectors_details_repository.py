from typing import Optional
from pydantic import BaseModel
from rocksdict import Rdict
from .base_documents_repository import BaseDocumentsRepository

class VectorIdentifier(BaseModel):
    value: str

class BaseVectorsDetailsRepository(BaseDocumentsRepository[VectorIdentifier]):
    
    def __init__(self, db_location: str, secondary: bool):
        super().__init__(db_location, secondary)

    def init_table(self, shard_name: str) -> None:
        self._init_table(shard_name)

    def drop_table(self, shard_name: str) -> None:
        self._drop_table(shard_name)
    
    def next_vector_id(self, shard_name: str) -> int:
        table: Rdict = self._get_table(shard_name)
        items = table.items(backwards=True)    
        try:
            last_key, _ = next(items)
            last_num = int(str(last_key).split(":")[-1])
            return last_num + 1
        except StopIteration:
            return 0
        
    def save_vector_ids(self, shard_name: str, embedding_id: str, vector_id: int) -> None:
        embedding_identifier: VectorIdentifier = VectorIdentifier(value=embedding_id)
        vector_identifier: VectorIdentifier = VectorIdentifier(value=str(vector_id))
        self._upsert_document(shard_name, embedding_id, vector_identifier)
        self._upsert_document(shard_name, str(vector_id), embedding_identifier)
    
    def delete_vector_ids(self, shard_name: str, embedding_id: str) -> None:
        vector_id: Optional[int] = self.read_vector_id(shard_name, embedding_id)
        if vector_id is not None:
            self._delete_document(shard_name, str(vector_id))
        self._delete_document(shard_name, embedding_id)

    def read_vector_id(self, shard_name: str, embedding_id: str) -> Optional[int]:
        vector_identifier: Optional[VectorIdentifier] = self._read_document(shard_name, embedding_id)
        if vector_identifier is None:
            return None
        return int(vector_identifier.value)
    
    def read_embedding_id(self, shard_name: str, vector_id: int) -> Optional[str]:
        embedding_identifier: Optional[VectorIdentifier] = self._read_document(shard_name, str(vector_id))
        if embedding_identifier is None:
            return None
        return embedding_identifier.value
    