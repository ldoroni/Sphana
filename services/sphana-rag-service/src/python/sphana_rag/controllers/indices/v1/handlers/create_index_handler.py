from injector import inject, singleton
from sphana_rag.controllers.indices.v1.schemas import CreateIndexRequest, CreateIndexResponse
from sphana_rag.services.indices import CreateIndexService
from request_handler import RequestHandler

@singleton
class CreateIndexHandler(RequestHandler[CreateIndexRequest, CreateIndexResponse]):

    @inject
    def __init__(self, 
                 create_index_service: CreateIndexService):
        super().__init__()
        self.__create_index_service = create_index_service

    async def _on_validate(self, request: CreateIndexRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: CreateIndexRequest) -> CreateIndexResponse:
        # Create index
        self.__create_index_service.create_index(
            index_name=request.index_name or "",
            description=request.description or "",
            number_of_shards=request.number_of_shards or 0,
            max_chunk_size=request.max_chunk_size or 0,
            chunk_overlap_size=request.chunk_overlap_size or 0
        )

        # Return response
        return CreateIndexResponse()
