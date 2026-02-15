from typing import Optional
from pydantic import BaseModel

class ListEntriesRequest(BaseModel):
    index_name: Optional[str] = None
    offset: Optional[str] = None
    limit: Optional[int] = None
    