from typing import Optional
from pydantic import BaseModel, Base64Bytes

class AppendPayloadRequest(BaseModel):
    index_name: Optional[str] = None
    entry_id: Optional[str] = None
    payload: Optional[Base64Bytes] = None
