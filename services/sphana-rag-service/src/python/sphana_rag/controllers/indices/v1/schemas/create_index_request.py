from typing import Optional
from pydantic import BaseModel

class CreateIndexRequest(BaseModel):
    index_name: Optional[str] = None
    description: Optional[str] = None
    number_of_shards: Optional[int] = None
    max_chunk_size: Optional[int] = None
    chunk_overlap_size: Optional[int] = None
