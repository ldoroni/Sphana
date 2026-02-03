from injector import inject, singleton
from sphana_rag.controllers.indices.v1.schemas import IndexExistsRequest, IndexExistsResponse
from sphana_rag.services.indices import IndexExistsService
from request_handler import RequestHandler

@singleton
class IndexExistsHandler(RequestHandler[IndexExistsRequest, IndexExistsResponse]):

    @inject
    def __init__(self, 
                 index_exists_service: IndexExistsService):
        super().__init__()
        self.__index_exists_service = index_exists_service

    async def _on_validate(self, request: IndexExistsRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: IndexExistsRequest) -> IndexExistsResponse:
        # Check if index exists
        exists: bool = self.__index_exists_service.index_exists(
            index_name=request.index_name or "",
        )

        # Return response
        return IndexExistsResponse(
            exists=exists
        )
