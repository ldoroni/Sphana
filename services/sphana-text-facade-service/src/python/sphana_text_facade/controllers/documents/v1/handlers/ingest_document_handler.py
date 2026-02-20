from injector import inject, singleton
from request_handler import RequestHandler
from sphana_text_facade.controllers.documents.v1.schemas import IngestDocumentRequest, IngestDocumentResponse
from sphana_text_facade.services.documents import IngestDocumentService

@singleton
class IngestDocumentHandler(RequestHandler[IngestDocumentRequest, IngestDocumentResponse]):

    @inject
    def __init__(self, ingest_document_service: IngestDocumentService) -> None:
        super().__init__()
        self._ingest_document_service = ingest_document_service

    def _on_validate(self, request: IngestDocumentRequest) -> None:
        pass

    def _on_invoke(self, request: IngestDocumentRequest) -> IngestDocumentResponse:
        # Ingest document
        self._ingest_document_service.ingest(
            index_name=request.index_name or "",
            entry_id=request.entry_id or "",
            title=request.title or "",
            content=request.content or "",
            metadata=request.metadata or {},
        )

        # Return response
        return IngestDocumentResponse()