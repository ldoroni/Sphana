from injector import inject, singleton
from sphana_data_store.controllers.indices.v1.schemas import GetIndexRequest, GetIndexResponse, IndexDetails
from sphana_data_store.services.indices import GetIndexService
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
                media_type=index_details.media_type,
                dimension=index_details.dimension,
                number_of_shards=index_details.number_of_shards,
                creation_timestamp=index_details.creation_timestamp,
                modification_timestamp=index_details.modification_timestamp
            )
        )