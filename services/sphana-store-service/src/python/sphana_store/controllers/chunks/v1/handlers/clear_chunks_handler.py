from injector import inject, singleton
from sphana_store.controllers.chunks.v1.schemas import ClearChunksRequest, ClearChunksResponse
from sphana_store.services.chunks import ClearChunksService
from request_handler import RequestHandler

@singleton
class ClearChunksHandler(RequestHandler[ClearChunksRequest, ClearChunksResponse]):
    
    @inject
    def __init__(self, 
                 clear_chunks_service: ClearChunksService):
        super().__init__()
        self.__clear_chunks_service = clear_chunks_service

    def _on_validate(self, request: ClearChunksRequest):
        # Validate request
        pass

    def _on_invoke(self, request: ClearChunksRequest) -> ClearChunksResponse:
        # Clear chunks
        self.__clear_chunks_service.clear_chunks(
            index_name=request.index_name or "",
            entry_id=request.entry_id or ""
        )

        # Return response
        return ClearChunksResponse()
