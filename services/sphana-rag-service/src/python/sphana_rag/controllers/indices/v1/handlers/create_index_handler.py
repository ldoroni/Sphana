from fastapi import Depends
from sphana_rag.controllers.indices.v1.schemas import CreateIndexRequest, CreateIndexResponse
from sphana_rag.services.indices import CreateIndexService
from request_handler import RequestHandler

class CreateIndexHandler(RequestHandler[CreateIndexRequest, CreateIndexResponse]):

    def __init__(self, 
                 create_index_service: CreateIndexService = Depends(CreateIndexService)):
        super().__init__()
        self._create_index_service = create_index_service

    async def _on_validate(self, request: CreateIndexRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: CreateIndexRequest) -> CreateIndexResponse:
        # Create index
        self._create_index_service.create_index(
            request.index_name or "",
            request.description or "",
            request.max_chunk_size or 0,
            request.max_chunk_overlap_size or 0
        )

        # Return response
        return CreateIndexResponse()
