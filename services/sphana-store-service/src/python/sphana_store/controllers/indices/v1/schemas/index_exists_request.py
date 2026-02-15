from typing import Optional
from pydantic import BaseModel

class IndexExistsRequest(BaseModel):
    index_name: Optional[str] = None
