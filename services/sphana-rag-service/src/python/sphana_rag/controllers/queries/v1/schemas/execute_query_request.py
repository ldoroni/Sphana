from typing import Optional
from pydantic import BaseModel

class ExecuteQueryRequest(BaseModel):
    index_name: Optional[str] = None
    query: Optional[str] = None
    max_results: Optional[int] = None
    score_threshold: Optional[float] = None
