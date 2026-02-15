from typing import Optional
from pydantic import BaseModel
from .chunk_details import ChunkDetails

class ChunkAndEmbedTextResponse(BaseModel):
    chunks: Optional[list[ChunkDetails]] = None
