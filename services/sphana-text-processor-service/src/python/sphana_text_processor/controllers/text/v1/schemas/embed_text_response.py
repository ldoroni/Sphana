from typing import Optional
from pydantic import BaseModel
from .embedding_details import EmbeddingDetails

class EmbedTextResponse(BaseModel):
    embedding: Optional[EmbeddingDetails] = None
