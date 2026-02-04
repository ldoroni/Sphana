from typing import Optional
from pydantic import BaseModel

class DeleteDocumentRequest(BaseModel):
    index_name: Optional[str] = None
    document_id: Optional[str] = None
    