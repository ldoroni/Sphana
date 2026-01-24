from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class DocumentDetails(BaseModel):
    # index_name: str
    document_id: str
    title: str
    content: str
    metadata: dict[str, str]
    creation_timestamp: datetime
    modification_timestamp: datetime

class IngestDocumentRequest(BaseModel):
    index_name: str
    document_id: str
    title: str
    content: str
    metadata: Optional[dict[str, str]]
    creation_timestamp: Optional[datetime]
    modification_timestamp: Optional[datetime]

class IngestDocumentResponse(BaseModel):
    pass

class UpdateDocumentRequest(BaseModel):
    index_name: str
    # document_id: str
    title: Optional[str]
    content: Optional[str]
    metadata: Optional[dict[str, str]]
    creation_timestamp: Optional[datetime]
    modification_timestamp: Optional[datetime]

class UpdateDocumentResponse(BaseModel):
    pass

class DeleteDocumentRequest(BaseModel):
    index_name: str
    document_id: str

class DeleteDocumentResponse(BaseModel):
    pass

class DocumentExistsRequest(BaseModel):
    index_name: str
    document_id: str

class DocumentExistsResponse(BaseModel):
    pass

class GetDocuentRequest(BaseModel):
    index_name: str
    document_id: str

class GetDocumentResponse(BaseModel):
    document: DocumentDetails

class ListDocumentsRequest(BaseModel):
    index_name: str
    offset: Optional[str]
    size: int

class ListDocumentsResponse(BaseModel):
    documents: list[DocumentDetails]
    offset: str
