from injector import inject, singleton
from sphana_rag.controllers.indices.v1.schemas import ListIndicesRequest, ListIndicesResponse, IndexDetails
from sphana_rag.services.indices import ListIndicesService
from request_handler import RequestHandler

@singleton
class ListIndicesHandler(RequestHandler[ListIndicesRequest, ListIndicesResponse]):

    @inject
    def __init__(self, 
                 list_indices_service: ListIndicesService):
        super().__init__()
        self.__list_indices_service = list_indices_service

    async def _on_validate(self, request: ListIndicesRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: ListIndicesRequest) -> ListIndicesResponse:
        # List indices
        results =self.__list_indices_service.list_indices(
            offset=request.offset,
            limit=request.limit or 0
        )

        # Return response
        return ListIndicesResponse(
            indices_details=[
                IndexDetails(
                    index_name=index.index_name,
                    description=index.description,
                    max_chunk_size=index.max_chunk_size,
                    chunk_overlap_size=index.chunk_overlap_size,
                    creation_timestamp=index.creation_timestamp,
                    modification_timestamp=index.modification_timestamp
                ) for index in results.documents
            ],
            next_offset=results.next_offset,
            completed=results.completed
        )
