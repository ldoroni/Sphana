from datetime import datetime
from pydantic import BaseModel

class ChunkDetails(BaseModel):
    chunk_id: str # <entry_id>:<chunk_id>
    entry_id: str # <entry_id>
    payload: bytes
    embedding_ids: list[str]
    creation_timestamp: datetime
