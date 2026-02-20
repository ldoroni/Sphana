from injector import inject, singleton
from sphana_data_store.controllers.embeddings.v1.schemas import ResetEmbeddingsRequest, ResetEmbeddingsResponse
from sphana_data_store.services.embeddings import ResetEmbeddingsService
from request_handler import RequestHandler

@singleton
class ResetEmbeddingsHandler(RequestHandler[ResetEmbeddingsRequest, ResetEmbeddingsResponse]):
    
    @inject
    def __init__(self, 
                 reset_embeddings_service: ResetEmbeddingsService):
        super().__init__()
        self.__reset_embeddings_service = reset_embeddings_service

    def _on_validate(self, request: ResetEmbeddingsRequest):
        # Validate request
        pass

    def _on_invoke(self, request: ResetEmbeddingsRequest) -> ResetEmbeddingsResponse:
        # Reset embeddings
        self.__reset_embeddings_service.reset_embeddings(
            index_name=request.index_name or "",
            entry_id=request.entry_id or ""
        )

        # Return response
        return ResetEmbeddingsResponse()
