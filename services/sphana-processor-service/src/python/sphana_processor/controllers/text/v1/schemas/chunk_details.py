from typing import Optional
from pydantic import BaseModel
from .embedding_details import EmbeddingDetails

class ChunkDetails(BaseModel):
    text: Optional[str] = None
    embeddings: Optional[list[EmbeddingDetails]] = None
