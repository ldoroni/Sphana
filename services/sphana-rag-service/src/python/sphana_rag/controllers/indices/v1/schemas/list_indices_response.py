from typing import Optional
from .index_details import IndexDetails
from pydantic import BaseModel

class ListIndicesResponse(BaseModel):
    indices: Optional[list[IndexDetails]] = None
    offset: Optional[str] = None
