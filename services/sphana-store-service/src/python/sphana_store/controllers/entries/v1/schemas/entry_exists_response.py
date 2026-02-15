from pydantic import BaseModel
from typing import Optional

class EntryExistsResponse(BaseModel):
    exists: Optional[bool] = None
