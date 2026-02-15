from typing import Optional
from pydantic import BaseModel

class ChunkAndEmbedTextRequest(BaseModel):
    text: Optional[str] = None
    max_parent_chunk_size: Optional[int] = None
    max_child_chunk_size: Optional[int] = None
    parent_chunk_overlap_size: Optional[int] = None
    child_chunk_overlap_size: Optional[int] = None
