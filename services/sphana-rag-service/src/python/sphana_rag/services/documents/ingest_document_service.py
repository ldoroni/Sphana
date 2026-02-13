from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException, ItemAlreadyExistsException
from sphana_rag.models import IndexDetails, DocumentDetails, ParentChunkDetails, ChildChunkDetails, TokenizedText
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ParentChunkDetailsRepository, ChildChunkDetailsRepository
from sphana_rag.services.cluster import ClusterRouterService
from sphana_rag.services.tokenizer import TextTokenizerService, TokenChunkerService, TextEmbedderService
from sphana_rag.utils import ShardUtil, CompressionUtil

TOPIC_INGEST_DOCUMENT = "shard.ingest_document"

@singleton
class IngestDocumentService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 parent_chunk_details_repository: ParentChunkDetailsRepository,
                 child_chunk_details_repository: ChildChunkDetailsRepository,
                 text_tokenizer_service: TextTokenizerService,
                 token_chunker_service: TokenChunkerService,
                 text_embedder_service: TextEmbedderService,
                 cluster_router_service: ClusterRouterService):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__text_tokenizer_service = text_tokenizer_service
        self.__token_chunker_service = token_chunker_service
        self.__text_embedder_service = text_embedder_service
        self.__cluster_router_service = cluster_router_service
        
        # Register listener for shard write operations
        self.__cluster_router_service.listen(TOPIC_INGEST_DOCUMENT, self._handle_ingest_writes)

    def ingest_document(self, index_name: str, document_id: str, title: str, content: str, metadata: dict[str, str]) -> None:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Get shard name
        shard_name: str = ShardUtil.compute_shard_name(
            index_name, 
            document_id, 
            index_details.number_of_shards
        )
        
        # Assert document does not already exist
        if self.__document_details_repository.exists(shard_name, document_id):
            raise ItemAlreadyExistsException(f"Document {document_id} already exists in index {index_name}")
        
        # Step 1: Tokenize the full document text
        tokenized_content: TokenizedText = self.__text_tokenizer_service.tokenize_text(content)
        
        # Step 2: Chunk tokens into parent chunks
        parent_chunks: list[TokenizedText] = self.__token_chunker_service.chunk_tokens(
            tokenized_text=tokenized_content,
            max_chunk_size=index_details.max_parent_chunk_size,
            chunk_overlap_size=index_details.parent_chunk_overlap_size
        )
        
        # Step 3: For each parent, chunk into child chunks
        child_texts: list[str] = [] # will be used for batch embedding (more performant than embedding one by one in the loop)
        child_texts_to_parent: list[int] = [] # map child index to parent index
        for parent_index, parent_chunk in enumerate(parent_chunks):
            child_chunks: list[TokenizedText] = self.__token_chunker_service.chunk_tokens(
                tokenized_text=parent_chunk,
                max_chunk_size=index_details.max_child_chunk_size,
                chunk_overlap_size=index_details.child_chunk_overlap_size
            )
            for child_chunk in child_chunks:
                child_texts.append(f"search_document: {child_chunk.text}")
                child_texts_to_parent.append(parent_index)
        
        # Step 4: Batch embed all child chunk texts
        child_embeddings: list[list[float]] = self.__text_embedder_service.embed_texts(child_texts)

        # Step 5: Aggregate child chunks by parent (list[tuple[parent index, single child embedding]])
        parent_child_embeddings: list[tuple[int, list[float]]] = []
        for i, parent_index in enumerate(child_texts_to_parent):
            parent_child_embeddings.append((parent_index, child_embeddings[i]))

        # Step 6: Route write operations to the shard owner
        message: dict = {
            "index_name": index_name,
            "document_id": document_id,
            "title": title,
            "content": content,
            "metadata": metadata,
            "parent_chunks": parent_chunks,
            "parent_child_embeddings": parent_child_embeddings
        }
        self.__cluster_router_service.route(shard_name, TOPIC_INGEST_DOCUMENT, message)

    def _handle_ingest_writes(self, shard_name: str, message: dict) -> Optional[dict]:
        # Get message payload
        index_name: str = message["index_name"]
        document_id: str = message["document_id"]
        title: str = message["title"]
        content: str = message["content"]
        metadata: dict[str, str] = message["metadata"]
        parent_chunks: list[TokenizedText] = message["parent_chunks"]
        parent_child_embeddings: list[tuple[int, list[float]]] = message["parent_child_embeddings"]

        # Assert document does not already exist
        if self.__document_details_repository.exists(shard_name, document_id):
            raise ItemAlreadyExistsException(f"Document {document_id} already exists in index {index_name}")
        
        # Step 7: Save parent chunks
        parent_chunk_ids: list[str] = []
        for parent_index, parent_chunk in enumerate(parent_chunks):
            parent_chunk_id: str = self.__parent_chunk_details_repository.next_unique_id(shard_name)
            parent_chunk_details: ParentChunkDetails = ParentChunkDetails(
                parent_chunk_id=parent_chunk_id,
                document_id=document_id,
                chunk_index=parent_index,
                content=CompressionUtil.compress(parent_chunk.text) # TODO: compress in the client before sending to reduce payload size?
            )
            self.__parent_chunk_details_repository.upsert(shard_name, parent_chunk_details)
            parent_chunk_ids.append(parent_chunk_id)
        
        # Step 8: Save child chunks with embeddings
        child_chunk_ids: list[str] = []
        for parent_index, child_embedding in parent_child_embeddings:
            parent_chunk_id = parent_chunk_ids[parent_index]
            child_chunk_id: str = self.__child_chunk_details_repository.next_unique_id(shard_name)
            child_chunk_details: ChildChunkDetails = ChildChunkDetails(
                child_chunk_id=child_chunk_id,
                parent_chunk_id=parent_chunk_id
            )
            self.__child_chunk_details_repository.upsert(shard_name, child_chunk_details)
            self.__index_vectors_repository.ingest(shard_name, child_chunk_id, child_embedding)
            child_chunk_ids.append(child_chunk_id)

        # Step 9: Save document details
        document_details: DocumentDetails = DocumentDetails(
            document_id=document_id,
            title=title,
            content=CompressionUtil.compress(content), # TODO: compress in the client before sending to reduce payload size?
            metadata=metadata,
            parent_chunk_ids=parent_chunk_ids,
            child_chunk_ids=child_chunk_ids,
            creation_timestamp=datetime.now(timezone.utc),
            modification_timestamp=datetime.now(timezone.utc)
        )
        self.__document_details_repository.upsert(shard_name, document_details)
        return None