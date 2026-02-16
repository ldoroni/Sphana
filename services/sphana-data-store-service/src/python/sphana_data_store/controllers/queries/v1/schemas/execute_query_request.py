from typing import Optional
from pydantic import BaseModel

class ExecuteQueryRequest(BaseModel):
    index_name: Optional[str] = None
    query_embedding: Optional[list[float]] = None
    max_results: Optional[int] = None
    score_threshold: Optional[float] = None
