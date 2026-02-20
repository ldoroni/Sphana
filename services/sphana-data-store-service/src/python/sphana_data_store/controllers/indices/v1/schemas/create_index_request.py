from typing import Optional
from pydantic import BaseModel

class CreateIndexRequest(BaseModel):
    index_name: Optional[str] = None
    description: Optional[str] = None
    media_type: Optional[str] = None
    dimension: Optional[int] = None
    number_of_shards: Optional[int] = None
