from datetime import datetime
from pydantic import BaseModel

class IndexDetails(BaseModel):
    index_name: str
    description: str
    max_chunk_size: int
    max_chunk_overlap_size: int
    creation_timestamp: datetime
    modification_timestamp: datetime