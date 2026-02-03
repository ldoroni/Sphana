from typing import Optional
from .index_details import IndexDetails
from pydantic import BaseModel

class GetIndexResponse(BaseModel):
    index_details: Optional[IndexDetails] = None
