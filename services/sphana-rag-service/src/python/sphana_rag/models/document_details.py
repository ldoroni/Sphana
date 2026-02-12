from datetime import datetime
from pydantic import BaseModel

class DocumentDetails(BaseModel):
    document_id: str
    title: str
    content: bytes
    metadata: dict[str, str]
    parent_chunk_ids: list[str]
    child_chunk_ids: list[str]
    creation_timestamp: datetime
    modification_timestamp: datetime