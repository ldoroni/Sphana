from typing import Optional
from pydantic import BaseModel

class DeleteIndexRequest(BaseModel):
    index_name: Optional[str] = None
