from typing import Optional
from pydantic import BaseModel
from .embedding_details import EmbeddingDetails

class AddEmbeddingsRequest(BaseModel):
    index_name: Optional[str] = None
    entry_id: Optional[str] = None
    embeddings: Optional[list[EmbeddingDetails]] = None
