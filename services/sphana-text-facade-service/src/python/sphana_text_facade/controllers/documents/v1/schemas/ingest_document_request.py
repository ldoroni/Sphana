from typing import Optional
from pydantic import BaseModel

class IngestDocumentRequest(BaseModel):
    index_name: Optional[str] = None
    entry_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
