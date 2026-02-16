from typing import Optional
from pydantic import BaseModel

class UpdateIndexRequest(BaseModel):
    index_name: Optional[str] = None
    description: Optional[str] = None
