from datetime import datetime
from pydantic import BaseModel

class EmbeddingDetails(BaseModel):
    embedding_id: str # 1,2,3...
    entry_id: str
    start_index: int
    end_index: int
    creation_timestamp: datetime
