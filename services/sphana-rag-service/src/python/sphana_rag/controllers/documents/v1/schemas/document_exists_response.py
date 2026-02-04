from pydantic import BaseModel
from typing import Optional

class DocumentExistsResponse(BaseModel):
    exists: Optional[bool] = None
