from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class IndexDetails(BaseModel):
    index_name: Optional[str] = None
    description: Optional[str] = None
    max_chunk_size: Optional[int] = None
    max_chunk_overlap_size: Optional[int] = None
    creation_timestamp: Optional[datetime] = None
    modification_timestamp: Optional[datetime] = None
