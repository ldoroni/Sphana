from injector import inject, singleton
from sphana_rag.controllers.documents.v1.schemas import DeleteDocumentRequest, DeleteDocumentResponse
from sphana_rag.services.documents import DeleteDocumentService
from request_handler import RequestHandler

@singleton
class DeleteDocumentHandler(RequestHandler[DeleteDocumentRequest, DeleteDocumentResponse]):

    @inject
    def __init__(self, 
                 ingest_document_service: DeleteDocumentService):
        super().__init__()
        self.__ingest_document_service = ingest_document_service

    async def _on_validate(self, request: DeleteDocumentRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: DeleteDocumentRequest) -> DeleteDocumentResponse:
        # Delete document
        self.__ingest_document_service.delete_document(
            index_name=request.index_name or "",
            document_id=request.document_id or ""
        )

        # Return response
        return DeleteDocumentResponse()
