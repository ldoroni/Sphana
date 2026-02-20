from typing import Optional
from pydantic import BaseModel

class AppendPayloadRequest(BaseModel):
    index_name: Optional[str] = None
    entry_id: Optional[str] = None
    payload: Optional[bytes] = None
