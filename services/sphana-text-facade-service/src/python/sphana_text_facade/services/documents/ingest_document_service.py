from injector import inject, singleton
from sphana_text_facade.models import TokenizedText
from sphana_text_facade.services.tokenizer import TextTokenizerService, TokenChunkerService, TextEmbedderService
from sphana_text_facade.clients.store import DataStoreEntriesClient, DataStorePayloadsClient, DataStoreEmbeddingsClient
from sphana_text_facade.clients.store.schemas import CreateEntryRequest, UploadPayloadRequest, AddEmbeddingsRequest, EmbeddingDetails
from sphana_text_facade.utils import Base64Util

@singleton
class IngestDocumentService:

    @inject
    def __init__(self,
                 text_tokenizer_service: TextTokenizerService,
                 token_chunker_service: TokenChunkerService,
                 text_embedder_service: TextEmbedderService,
                 data_store_entries_client: DataStoreEntriesClient,
                 data_store_payloads_client: DataStorePayloadsClient,
                 data_store_embeddings_client: DataStoreEmbeddingsClient) -> None:
        self.__text_tokenizer_service = text_tokenizer_service
        self.__token_chunker_service = token_chunker_service
        self.__text_embedder_service = text_embedder_service
        self.__data_store_entries_client = data_store_entries_client
        self.__data_store_payloads_client = data_store_payloads_client
        self.__data_store_embeddings_client = data_store_embeddings_client

    def ingest(self, index_name: str, entry_id: str, title: str, content: str,  metadata: dict[str, str]) -> None:
        # Step 1: Send text to processor for chunking and embedding
        # TODO: take parameters from index details!
        embeddings: list[EmbeddingDetails] = self.__chunk_and_embed_text(
            text=content,
            max_parent_chunk_size=4000,
            max_child_chunk_size=200,
            parent_chunk_overlap_size=1000,
            child_chunk_overlap_size=50
        )

        # Step 2: Create entry in data store
        create_entry_request: CreateEntryRequest = CreateEntryRequest(
            index_name=index_name,
            entry_id=entry_id,
            title=title,
            metadata=metadata,
        )
        self.__data_store_entries_client.create_entry(create_entry_request)

        # Step 3: Upload payload to data store
        upload_payload_request: UploadPayloadRequest = UploadPayloadRequest(
            index_name=index_name,
            entry_id=entry_id,
            payload=Base64Util.to_nullable_base64(content),
        )
        self.__data_store_payloads_client.upload_payload(upload_payload_request)

        # Step 4: Add embeddings to data store
        add_embeddings_request: AddEmbeddingsRequest = AddEmbeddingsRequest(
            index_name=index_name,
            entry_id=entry_id,
            embeddings=embeddings
        )
        self.__data_store_embeddings_client.add_embeddings(add_embeddings_request)

    def __chunk_and_embed_text(self, text: str, max_parent_chunk_size: int, max_child_chunk_size: int, parent_chunk_overlap_size: int, child_chunk_overlap_size: int) -> list[EmbeddingDetails]:
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

        # Step 5: Get embeddings details
        embeddings: list[EmbeddingDetails] = []
        for child_index, parent_index in enumerate(child_texts_to_parent):
            embedding_details: EmbeddingDetails = EmbeddingDetails(
                start_index=parent_chunks[parent_index].start_char,
                end_index=parent_chunks[parent_index].end_char,
                embedding=child_embeddings[child_index]
            )
            embeddings.append(embedding_details)
        return embeddings