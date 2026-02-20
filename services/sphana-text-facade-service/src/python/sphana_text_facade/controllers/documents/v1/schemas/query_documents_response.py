from typing import Optional
from pydantic import BaseModel
from .query_documents_result import QueryDocumentsResult

class QueryDocumentsResponse(BaseModel):
    results: Optional[list[QueryDocumentsResult]] = None