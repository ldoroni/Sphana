from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, DocumentDetails, ParentChunkDetails, ChildChunkDetails, TokenizedText
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ParentChunkDetailsRepository, ChildChunkDetailsRepository, DistributedCacheRepository
from sphana_rag.services.tokenizer import TextTokenizerService, TokenChunkerService, TextEmbedderService
from sphana_rag.utils import ShardUtil, CompressionUtil

@singleton
class UpdateDocumentService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 parent_chunk_details_repository: ParentChunkDetailsRepository,
                 child_chunk_details_repository: ChildChunkDetailsRepository,
                 distributed_cache_repository: DistributedCacheRepository,
                 text_tokenizer_service: TextTokenizerService,
                 token_chunker_service: TokenChunkerService,
                 text_embedder_service: TextEmbedderService):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__distributed_cache_repository = distributed_cache_repository
        self.__text_tokenizer_service = text_tokenizer_service
        self.__token_chunker_service = token_chunker_service
        self.__text_embedder_service = text_embedder_service

    def update_document(self, index_name: str, document_id: str, title: Optional[str], content: Optional[str], metadata: Optional[dict[str, str]]):
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
        if not self.__document_details_repository.exists(shard_name, document_id):
            raise ItemNotFoundException(f"Document {document_id} does not exist in index {index_name}")
        
        # Chunk content
        if content is not None:
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
            child_chunk_embeddings: list[tuple[int, list[float]]] = []
            for i, parent_index in enumerate(child_texts_to_parent):
                child_chunk_embeddings.append((parent_index, child_embeddings[i]))
        else:
            parent_chunks = []
            child_chunk_embeddings = []

        # Save information in the repository
        with self.__distributed_cache_repository.lock(shard_name, ttl_seconds=300):
            # Get document details
            document_details: Optional[DocumentDetails] = self.__document_details_repository.read(shard_name, document_id)
            if document_details is None:
                raise ItemNotFoundException(f"Document {document_id} does not exist in index {index_name}")
            
            # Update content and chunks
            if content is not None:
                # Step 7: Delete old parent chunks
                for parent_chunk_id in document_details.parent_chunk_ids:
                    self.__parent_chunk_details_repository.delete(shard_name, parent_chunk_id)

                # Step 8: Delete old child chunks and their vectors
                for child_chunk_id in document_details.child_chunk_ids:
                    self.__child_chunk_details_repository.delete(shard_name, child_chunk_id)
                    self.__index_vectors_repository.delete(shard_name, child_chunk_id)

                # Step 9: Save parent chunks
                parent_chunk_ids: list[str] = []
                for parent_index, parent_chunk in enumerate(parent_chunks):
                    parent_chunk_id: str = self.__parent_chunk_details_repository.next_unique_id(shard_name)
                    parent_chunk_details: ParentChunkDetails = ParentChunkDetails(
                        parent_chunk_id=parent_chunk_id,
                        document_id=document_id,
                        chunk_index=parent_index,
                        content=CompressionUtil.compress(parent_chunk.text)
                    )
                    self.__parent_chunk_details_repository.upsert(shard_name, parent_chunk_details)
                    parent_chunk_ids.append(parent_chunk_id)
                
                # Step 10: Save child chunks with embeddings
                child_chunk_ids: list[str] = []
                for parent_index, child_embedding in child_chunk_embeddings:
                    parent_chunk_id = parent_chunk_ids[parent_index]
                    child_chunk_id: str = self.__child_chunk_details_repository.next_unique_id(shard_name)
                    child_chunk_details: ChildChunkDetails = ChildChunkDetails(
                        child_chunk_id=child_chunk_id,
                        parent_chunk_id=parent_chunk_id
                    )
                    self.__child_chunk_details_repository.upsert(shard_name, child_chunk_details)
                    self.__index_vectors_repository.ingest(shard_name, child_chunk_id, child_embedding)
                    child_chunk_ids.append(child_chunk_id)
                
                # Step 11: Update document details with new content, parent chunk ids and child chunk ids
                document_details.content = CompressionUtil.compress(content)
                document_details.parent_chunk_ids = parent_chunk_ids
                document_details.child_chunk_ids = child_chunk_ids

            # Update title
            if title is not None:
                document_details.title = title
            
            # Update metadata
            if metadata is not None:
                document_details.metadata = metadata
            
            # Update modification timestamp
            document_details.modification_timestamp = datetime.now(timezone.utc)

            # Step 12: Save document details
            self.__document_details_repository.upsert(shard_name, document_details)