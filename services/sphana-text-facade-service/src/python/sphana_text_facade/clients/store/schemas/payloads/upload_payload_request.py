from typing import Optional
from pydantic import BaseModel, Base64Str

class UploadPayloadRequest(BaseModel):
    index_name: Optional[str] = None
    entry_id: Optional[str] = None
    payload: Optional[Base64Str] = None
