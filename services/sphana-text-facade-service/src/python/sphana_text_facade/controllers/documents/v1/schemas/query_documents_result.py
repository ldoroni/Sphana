from typing import Optional
from pydantic import BaseModel

class QueryDocumentsResult(BaseModel):
    entry_id: Optional[str] = None
    payload: Optional[str] = None
    score: Optional[float] = None