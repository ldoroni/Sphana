from typing import Optional
from pydantic import BaseModel

class CreateEntryRequest(BaseModel):
    index_name: Optional[str] = None
    entry_id: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
