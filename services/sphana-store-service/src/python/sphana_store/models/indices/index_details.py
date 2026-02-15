from datetime import datetime
from pydantic import BaseModel

class IndexDetails(BaseModel):
    index_name: str
    description: str
    number_of_shards: int
    max_parent_chunk_size: int
    max_child_chunk_size: int
    parent_chunk_overlap_size: int
    child_chunk_overlap_size: int
    creation_timestamp: datetime
    modification_timestamp: datetime