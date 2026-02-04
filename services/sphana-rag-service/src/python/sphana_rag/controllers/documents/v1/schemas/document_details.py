from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class DocumentDetails(BaseModel):
    document_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    creation_timestamp: Optional[datetime] = None
    modification_timestamp: Optional[datetime] = None
    