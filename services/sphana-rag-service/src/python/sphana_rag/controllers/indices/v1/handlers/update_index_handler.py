from injector import inject, singleton
from sphana_rag.controllers.indices.v1.schemas import UpdateIndexRequest, UpdateIndexResponse
from sphana_rag.services.indices import UpdateIndexService
from request_handler import RequestHandler

@singleton
class UpdateIndexHandler(RequestHandler[UpdateIndexRequest, UpdateIndexResponse]):

    @inject
    def __init__(self, 
                 update_index_service: UpdateIndexService):
        super().__init__()
        self.__update_index_service = update_index_service

    async def _on_validate(self, request: UpdateIndexRequest):
        # Validate request
        pass

    async def _on_invoke(self, request: UpdateIndexRequest) -> UpdateIndexResponse:
        # Update index
        self.__update_index_service.update_index(
            index_name=request.index_name or "",
            description=request.description or ""
        )

        # Return response
        return UpdateIndexResponse()
