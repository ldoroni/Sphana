from datetime import datetime
from pydantic import BaseModel

class DocumentDetails(BaseModel):
    document_id: str
    title: str
    content: str
    metadata: dict[str, str]
    chunk_ids: list[str]
    creation_timestamp: datetime
    modification_timestamp: datetime