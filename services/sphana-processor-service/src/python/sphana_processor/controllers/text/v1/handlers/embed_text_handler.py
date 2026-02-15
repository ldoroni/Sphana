from injector import inject, singleton
from sphana_processor.controllers.text.v1.schemas import EmbedTextRequest, EmbedTextResponse, EmbeddingDetails
from sphana_processor.services.text import TextEmbedderService
from request_handler import RequestHandler

@singleton
class EmbedTextHandler(RequestHandler[EmbedTextRequest, EmbedTextResponse]):

    @inject
    def __init__(self, 
                 text_embedder_service: TextEmbedderService):
        super().__init__()
        self.__text_embedder_service = text_embedder_service

    def _on_validate(self, request: EmbedTextRequest):
        # Validate request
        pass

    def _on_invoke(self, request: EmbedTextRequest) -> EmbedTextResponse:
        # Embed text
        embedding: list[float] = self.__text_embedder_service.embed_text(
            text=request.text or ""
        )

        # Return response
        return EmbedTextResponse(
            embedding=EmbeddingDetails(embedding=embedding)
        )
