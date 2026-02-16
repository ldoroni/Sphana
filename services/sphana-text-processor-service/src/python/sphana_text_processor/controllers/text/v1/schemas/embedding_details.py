from typing import Optional
from pydantic import BaseModel

class EmbeddingDetails(BaseModel):
    embedding: Optional[list[float]] = None
