from typing import Optional
from pydantic import BaseModel

class UpdateDocumentRequest(BaseModel):
    index_name: Optional[str] = None
    document_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    