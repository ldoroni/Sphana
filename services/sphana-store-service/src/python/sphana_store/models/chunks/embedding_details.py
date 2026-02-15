from datetime import datetime
from pydantic import BaseModel

class EmbeddingDetails(BaseModel):
    embedding_id: str # 1,2,3...
    chunk_id: str # <entry_id>:<chunk_id>
    creation_timestamp: datetime
