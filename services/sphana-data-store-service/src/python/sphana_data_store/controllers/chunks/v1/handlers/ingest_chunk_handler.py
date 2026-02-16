from injector import inject, singleton
from sphana_data_store.controllers.chunks.v1.schemas import IngestChunkRequest, IngestChunkResponse
from sphana_data_store.services.chunks import IngestChunkService
from request_handler import RequestHandler

@singleton
class IngestChunkHandler(RequestHandler[IngestChunkRequest, IngestChunkResponse]):
    
    @inject
    def __init__(self, 
                 ingest_chunk_service: IngestChunkService):
        super().__init__()
        self.__ingest_chunk_service = ingest_chunk_service

    def _on_validate(self, request: IngestChunkRequest):
        # Validate request
        pass

    def _on_invoke(self, request: IngestChunkRequest) -> IngestChunkResponse:
        # Ingest chunk
        self.__ingest_chunk_service.ingest_chunk(
            index_name=request.index_name or "",
            entry_id=request.entry_id or "",
            payload=request.payload or b"",
            embeddings=[r.embedding for r in request.embeddings or [] if r.embedding is not None]
        )

        # Return response
        return IngestChunkResponse()
