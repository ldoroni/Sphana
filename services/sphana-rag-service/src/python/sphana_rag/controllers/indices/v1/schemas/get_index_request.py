from typing import Optional
from pydantic import BaseModel

class GetIndexRequest(BaseModel):
    index_name: Optional[str] = None
