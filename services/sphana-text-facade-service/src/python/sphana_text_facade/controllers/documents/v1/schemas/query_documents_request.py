from typing import Optional
from pydantic import BaseModel

class QueryDocumentsRequest(BaseModel):
    index_name: Optional[str] = None
    query_text: Optional[str] = None
    max_results: Optional[int] = None
    score_threshold: Optional[float] = None