from injector import inject, singleton
from sphana_data_store.controllers.indices.v1.schemas import CreateIndexRequest, CreateIndexResponse
from sphana_data_store.services.indices import CreateIndexService
from request_handler import RequestHandler

@singleton
class CreateIndexHandler(RequestHandler[CreateIndexRequest, CreateIndexResponse]):

    @inject
    def __init__(self, 
                 create_index_service: CreateIndexService):
        super().__init__()
        self.__create_index_service = create_index_service

    def _on_validate(self, request: CreateIndexRequest):
        # Validate request
        pass

    def _on_invoke(self, request: CreateIndexRequest) -> CreateIndexResponse:
        # Create index
        self.__create_index_service.create_index(
            index_name=request.index_name or "",
            description=request.description or "",
            number_of_shards=request.number_of_shards or 0
        )

        # Return response
        return CreateIndexResponse()
