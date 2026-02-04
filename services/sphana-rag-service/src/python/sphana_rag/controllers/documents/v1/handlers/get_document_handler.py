from injector import inject, singleton
from sphana_rag.controllers.documents.v1.schemas import GetDocumentRequest, GetDocumentResponse, DocumentDetails
from sphana_rag.services.documents import GetDocumentService
from request_handler import RequestHandler

@singleton
class GetDocumentHandler(RequestHandler[GetDocumentRequest, GetDocumentResponse]):

    @inject
    def __init__(self, 
                 get_document_service: GetDocumentService):
        super().__init__()
        self.__get_document_service = get_document_service

    async def _on_validate(self, request: GetDocumentRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: GetDocumentRequest) -> GetDocumentResponse:
        # Delete document
        document_details = self.__get_document_service.get_document(
            index_name=request.index_name or "",
            document_id=request.document_id or ""
        )

        # Return response
        return GetDocumentResponse(
            document_details=DocumentDetails(
                document_id=document_details.document_id,
                title=document_details.title,
                content=document_details.content,
                metadata=document_details.metadata,
                creation_timestamp=document_details.creation_timestamp,
                modification_timestamp=document_details.modification_timestamp
            )
        )
