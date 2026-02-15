from typing import Optional
from pydantic import BaseModel

class CreateIndexRequest(BaseModel):
    index_name: Optional[str] = None
    description: Optional[str] = None
    number_of_shards: Optional[int] = None
    max_parent_chunk_size: Optional[int] = None
    max_child_chunk_size: Optional[int] = None
    parent_chunk_overlap_size: Optional[int] = None
    child_chunk_overlap_size: Optional[int] = None
