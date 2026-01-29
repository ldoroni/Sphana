from datetime import datetime
from pydantic import BaseModel

class ChunkDetails(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    creation_timestamp: datetime