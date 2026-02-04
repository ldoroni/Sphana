from typing import Optional
from pydantic import BaseModel
from .document_details import DocumentDetails

class ListDocumentsResponse(BaseModel):
    documents_details: Optional[list[DocumentDetails]] = None
    next_offset: Optional[str] = None
    completed: Optional[bool] = None
