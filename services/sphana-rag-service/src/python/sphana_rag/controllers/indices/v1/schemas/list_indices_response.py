from typing import Optional
from .index_details import IndexDetails
from pydantic import BaseModel

class ListIndicesResponse(BaseModel):
    indices_details: Optional[list[IndexDetails]] = None
    next_offset: Optional[str] = None
    completed: Optional[bool] = None
