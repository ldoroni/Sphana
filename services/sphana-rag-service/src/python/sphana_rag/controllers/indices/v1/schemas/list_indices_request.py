from typing import Optional
from pydantic import BaseModel

class ListIndicesRequest(BaseModel):
    offset: Optional[str] = None
    size: Optional[int] = None
