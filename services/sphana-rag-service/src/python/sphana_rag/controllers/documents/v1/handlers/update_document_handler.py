from injector import inject, singleton
from sphana_rag.controllers.documents.v1.schemas import UpdateDocumentRequest, UpdateDocumentResponse
from sphana_rag.services.documents import UpdateDocumentService
from request_handler import RequestHandler

@singleton
class UpdateDocumentHandler(RequestHandler[UpdateDocumentRequest, UpdateDocumentResponse]):

    @inject
    def __init__(self, 
                 ingest_document_service: UpdateDocumentService):
        super().__init__()
        self.__ingest_document_service = ingest_document_service

    async def _on_validate(self, request: UpdateDocumentRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: UpdateDocumentRequest) -> UpdateDocumentResponse:
        # Ingest document
        self.__ingest_document_service.update_document(
            index_name=request.index_name or "",
            document_id=request.document_id or "",
            title=request.title,
            content=request.content,
            metadata=request.metadata
        )

        # Return response
        return UpdateDocumentResponse()
