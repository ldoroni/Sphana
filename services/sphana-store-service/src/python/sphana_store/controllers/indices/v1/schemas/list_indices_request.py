from typing import Optional
from pydantic import BaseModel

class ListIndicesRequest(BaseModel):
    offset: Optional[str] = None
    limit: Optional[int] = None
