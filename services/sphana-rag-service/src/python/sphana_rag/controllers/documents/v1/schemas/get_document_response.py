from typing import Optional
from pydantic import BaseModel
from .document_details import DocumentDetails

class GetDocumentResponse(BaseModel):
    document_details: Optional[DocumentDetails] = None
