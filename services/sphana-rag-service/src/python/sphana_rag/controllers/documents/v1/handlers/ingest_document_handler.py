from fastapi import Depends
from sphana_rag.controllers.documents.v1.schemas import IngestDocumentRequest, IngestDocumentResponse
from sphana_rag.services.documents import IngestDocumentService
from request_handler import RequestHandler

class IngestDocumentHandler(RequestHandler[IngestDocumentRequest, IngestDocumentResponse]):

    def __init__(self, 
                 ingest_document_service: IngestDocumentService = Depends(IngestDocumentService)):
        super().__init__()
        self.__ingest_document_service = ingest_document_service

    async def _on_validate(self, request: IngestDocumentRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: IngestDocumentRequest) -> IngestDocumentResponse:
        # Ingest document
        self.__ingest_document_service.ingest_document(
            index_name=request.index_name or "",
            document_id=request.document_id or "",
            title=request.title or "",
            content=request.content or "",
            metadata=request.metadata or {}
        )

        # Return response
        return IngestDocumentResponse()
