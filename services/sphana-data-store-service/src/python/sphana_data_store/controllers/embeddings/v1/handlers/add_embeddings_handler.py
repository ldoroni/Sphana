from injector import inject, singleton
from sphana_data_store.controllers.embeddings.v1.schemas import AddEmbeddingsRequest, AddEmbeddingsResponse
from sphana_data_store.services.embeddings import AddEmbeddingsService
from request_handler import RequestHandler

@singleton
class AddEmbeddingsHandler(RequestHandler[AddEmbeddingsRequest, AddEmbeddingsResponse]):
    
    @inject
    def __init__(self, 
                 add_embeddings_service: AddEmbeddingsService):
        super().__init__()
        self.__add_embeddings_service = add_embeddings_service

    def _on_validate(self, request: AddEmbeddingsRequest):
        # Validate request
        pass

    def _on_invoke(self, request: AddEmbeddingsRequest) -> AddEmbeddingsResponse:
        # Ingest chunk
        self.__add_embeddings_service.add_embeddings(
            index_name=request.index_name or "",
            entry_id=request.entry_id or "",
            start_indexes=[r.start_index for r in request.embeddings or [] if r.start_index is not None],
            end_indexes=[r.end_index for r in request.embeddings or [] if r.end_index is not None],
            embeddings=[r.embedding for r in request.embeddings or [] if r.embedding is not None]
        )

        # Return response
        return AddEmbeddingsResponse()
