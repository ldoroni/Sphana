from injector import inject, singleton
from sphana_rag.controllers.documents.v1.schemas import DocumentExistsRequest, DocumentExistsResponse
from sphana_rag.services.documents import DocumentExistsService
from request_handler import RequestHandler

@singleton
class DocumentExistsHandler(RequestHandler[DocumentExistsRequest, DocumentExistsResponse]):

    @inject
    def __init__(self, 
                 document_exists_service: DocumentExistsService):
        super().__init__()
        self.__document_exists_service = document_exists_service

    def _on_validate(self, request: DocumentExistsRequest):
        # Validate request
        pass

    def _on_invoke(self, request: DocumentExistsRequest) -> DocumentExistsResponse:
        # Check if document exists
        exists: bool = self.__document_exists_service.document_exists(
            index_name=request.index_name or "",
            document_id=request.document_id or ""
        )

        # Return response
        return DocumentExistsResponse(
            exists=exists
        )
