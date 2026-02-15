from injector import inject, singleton
from sphana_processor.controllers.text.v1.schemas import ChunkAndEmbedTextRequest, ChunkAndEmbedTextResponse, ChunkDetails, EmbeddingDetails
from sphana_processor.models import TokenizedText
from sphana_processor.services.text import TextEmbedderService, TextTokenizerService, TokenChunkerService
from request_handler import RequestHandler

@singleton
class ChunkAndEmbedTextHandler(RequestHandler[ChunkAndEmbedTextRequest, ChunkAndEmbedTextResponse]):

    @inject
    def __init__(self, 
                 text_tokenizer_service: TextTokenizerService,
                 token_chunker_service: TokenChunkerService,
                 text_embedder_service: TextEmbedderService):
        super().__init__()
        self.__text_tokenizer_service = text_tokenizer_service
        self.__token_chunker_service = token_chunker_service
        self.__text_embedder_service = text_embedder_service

    def _on_validate(self, request: ChunkAndEmbedTextRequest):
        # Validate request
        pass

    def _on_invoke(self, request: ChunkAndEmbedTextRequest) -> ChunkAndEmbedTextResponse:
        text: str = request.text or ""
        max_parent_chunk_size: int = request.max_parent_chunk_size or 512
        max_child_chunk_size: int = request.max_child_chunk_size or 128
        parent_chunk_overlap_size: int = request.parent_chunk_overlap_size or 50
        child_chunk_overlap_size: int = request.child_chunk_overlap_size or 20
        
        # Step 1: Tokenize the full document text
        tokenized_content: TokenizedText = self.__text_tokenizer_service.tokenize_text(text)
        
        # Step 2: Chunk tokens into parent chunks
        parent_chunks: list[TokenizedText] = self.__token_chunker_service.chunk_tokens(
            tokenized_text=tokenized_content,
            max_chunk_size=max_parent_chunk_size,
            chunk_overlap_size=parent_chunk_overlap_size
        )
        
        # Step 3: For each parent, chunk into child chunks
        child_texts: list[str] = [] # will be used for batch embedding (more performant than embedding one by one in the loop)
        child_texts_to_parent: list[int] = [] # map child index to parent index
        for parent_index, parent_chunk in enumerate(parent_chunks):
            child_chunks: list[TokenizedText] = self.__token_chunker_service.chunk_tokens(
                tokenized_text=parent_chunk,
                max_chunk_size=max_child_chunk_size,
                chunk_overlap_size=child_chunk_overlap_size
            )
            for child_chunk in child_chunks:
                child_texts.append(f"search_document: {child_chunk.text}")
                child_texts_to_parent.append(parent_index)
        
        # Step 4: Batch embed all child chunk texts
        child_embeddings: list[list[float]] = self.__text_embedder_service.embed_texts(child_texts)

        # Step 5: Aggregate child chunks by parent (list[tuple[parent index, single child embedding]])
        child_chunk_embeddings: dict[int, list[list[float]]] = {}
        for child_index, parent_index in enumerate(child_texts_to_parent):
            if parent_index not in child_chunk_embeddings:
                child_chunk_embeddings[parent_index] = []
            child_chunk_embeddings[parent_index].append(child_embeddings[child_index])

        # Return response
        return ChunkAndEmbedTextResponse(
            chunks=[
                ChunkDetails(
                    text=parent_chunks[parent_index].text,
                    embeddings=[EmbeddingDetails(embedding=embedding) for embedding in child_chunk_embeddings.get(parent_index, [])]
                )
                for parent_index in range(len(parent_chunks))
            ]
        )
