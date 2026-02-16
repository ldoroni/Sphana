from typing import Optional
from pydantic import BaseModel
from .embedding_details import EmbeddingDetails

class IngestChunkRequest(BaseModel):
    index_name: Optional[str] = None
    entry_id: Optional[str] = None
    payload: Optional[bytes] = None
    embeddings: Optional[list[EmbeddingDetails]] = None
