from typing import Optional
from pydantic import BaseModel

class EntryExistsRequest(BaseModel):
    index_name: Optional[str] = None
    entry_id: Optional[str] = None
    