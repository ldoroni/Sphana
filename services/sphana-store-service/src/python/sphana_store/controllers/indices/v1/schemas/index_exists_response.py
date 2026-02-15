from pydantic import BaseModel
from typing import Optional

class IndexExistsResponse(BaseModel):
    exists: Optional[bool] = None
