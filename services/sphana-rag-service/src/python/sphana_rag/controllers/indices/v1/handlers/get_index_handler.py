from injector import inject, singleton
from sphana_rag.controllers.indices.v1.schemas import GetIndexRequest, GetIndexResponse, IndexDetails
from sphana_rag.services.indices import GetIndexService
from request_handler import RequestHandler

@singleton
class GetIndexHandler(RequestHandler[GetIndexRequest, GetIndexResponse]):

    @inject
    def __init__(self, 
                 get_index_service: GetIndexService):
        super().__init__()
        self.__get_index_service = get_index_service

    def _on_validate(self, request: GetIndexRequest):
        # Validate request
        pass

    def _on_invoke(self, request: GetIndexRequest) -> GetIndexResponse:
        # Get index
        index_details = self.__get_index_service.get_index(
            index_name=request.index_name or "",
        )

        # Return response
        return GetIndexResponse(
            index_details=IndexDetails(
                index_name=index_details.index_name,
                description=index_details.description,
                number_of_shards=index_details.number_of_shards,
                max_parent_chunk_size=index_details.max_parent_chunk_size,
                max_child_chunk_size=index_details.max_child_chunk_size,
                parent_chunk_overlap_size=index_details.parent_chunk_overlap_size,
                child_chunk_overlap_size=index_details.child_chunk_overlap_size,
                creation_timestamp=index_details.creation_timestamp,
                modification_timestamp=index_details.modification_timestamp
            )
        )