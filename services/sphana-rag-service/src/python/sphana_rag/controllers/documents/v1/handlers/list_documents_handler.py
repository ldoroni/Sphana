from injector import inject, singleton
from sphana_rag.controllers.documents.v1.schemas import ListDocumentsRequest, ListDocumentsResponse, DocumentDetails
from sphana_rag.services.documents import ListDocumentsService
from sphana_rag.utils import CompressionUtil
from request_handler import RequestHandler

@singleton
class ListDocumentsHandler(RequestHandler[ListDocumentsRequest, ListDocumentsResponse]):

    @inject
    def __init__(self, 
                 list_documents_service: ListDocumentsService):
        super().__init__()
        self.__list_documents_service = list_documents_service

    async def _on_validate(self, request: ListDocumentsRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: ListDocumentsRequest) -> ListDocumentsResponse:
        # List documents
        document_details = self.__list_documents_service.list_documents(
            index_name=request.index_name or "",
            offset=request.offset,
            limit=request.limit or 0
        )

        # Return response
        return ListDocumentsResponse(
            documents_details=[
                DocumentDetails(
                    document_id=document.document_id,
                    title=document.title,
                    content=CompressionUtil.decompress(document.content), # TODO: I dislike the decompression here!
                    metadata=document.metadata,
                    creation_timestamp=document.creation_timestamp,
                    modification_timestamp=document.modification_timestamp
                ) for document in document_details.documents
            ],
            next_offset=document_details.next_offset,
            completed=document_details.completed
        )
