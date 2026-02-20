from typing import Optional
from pydantic import BaseModel

class EmbeddingDetails(BaseModel):
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    embedding: Optional[list[float]] = None
    