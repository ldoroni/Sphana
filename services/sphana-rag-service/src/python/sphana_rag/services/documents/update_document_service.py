from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, DocumentDetails, ParentChunkDetails, ChildChunkDetails, TokenizedText
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ParentChunkDetailsRepository, ChildChunkDetailsRepository
from sphana_rag.services.tokenizer import TextTokenizer, TokenChunker, TextEmbedder
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
                 text_tokenizer: TextTokenizer,
                 token_chunker: TokenChunker,
                 text_embedder: TextEmbedder):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__text_tokenizer = text_tokenizer
        self.__token_chunker = token_chunker
        self.__text_embedder = text_embedder

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
        
        # Get document details
        document_details: Optional[DocumentDetails] = self.__document_details_repository.read(shard_name, document_id)
        if document_details is None:
            raise ItemNotFoundException(f"Document {document_id} does not exist in index {index_name}")
        
        if content is not None:
            # Delete old parent chunks
            for parent_chunk_id in document_details.parent_chunk_ids:
                self.__parent_chunk_details_repository.delete(shard_name, parent_chunk_id)

            # Delete old child chunks and their vectors
            for child_chunk_id in document_details.child_chunk_ids:
                self.__child_chunk_details_repository.delete(shard_name, child_chunk_id)
                self.__index_vectors_repository.delete(shard_name, child_chunk_id)

            # Step 1: Tokenize the full document text
            tokenized_content: TokenizedText = self.__text_tokenizer.tokenize_text(content)
            
            # Step 2: Chunk tokens into parent chunks
            parent_chunks: list[TokenizedText] = self.__token_chunker.chunk_tokens(
                tokenized_text=tokenized_content,
                max_chunk_size=index_details.max_parent_chunk_size,
                chunk_overlap_size=index_details.parent_chunk_overlap_size
            )
            
            # Step 3: For each parent, chunk into child chunks
            parent_child_map: list[tuple[int, TokenizedText]] = []
            child_texts: list[str] = []
            
            for parent_index, parent_chunk in enumerate(parent_chunks):
                child_chunks: list[TokenizedText] = self.__token_chunker.chunk_tokens(
                    tokenized_text=parent_chunk,
                    max_chunk_size=index_details.max_child_chunk_size,
                    chunk_overlap_size=index_details.child_chunk_overlap_size
                )
                for child_chunk in child_chunks:
                    parent_child_map.append((parent_index, child_chunk))
                    child_texts.append(f"search_document: {child_chunk.text}")
            
            # Step 4: Batch embed all child chunk texts
            embeddings: list[list[float]] = self.__text_embedder.embed_texts(child_texts)

            # Step 5: Save new parent chunks
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
            
            # Step 6: Save new child chunks with embeddings
            child_chunk_ids: list[str] = []
            for embedding_index, (parent_index, child_chunk) in enumerate(parent_child_map):
                parent_chunk_id = parent_chunk_ids[parent_index]
                child_chunk_id: str = self.__child_chunk_details_repository.next_unique_id(shard_name)
                child_chunk_details: ChildChunkDetails = ChildChunkDetails(
                    child_chunk_id=child_chunk_id,
                    parent_chunk_id=parent_chunk_id
                )
                self.__child_chunk_details_repository.upsert(shard_name, child_chunk_details)
                self.__index_vectors_repository.ingest(shard_name, child_chunk_id, embeddings[embedding_index])
                child_chunk_ids.append(child_chunk_id)

            # Update document details with new chunks
            document_details.content = CompressionUtil.compress(content)
            document_details.parent_chunk_ids = parent_chunk_ids
            document_details.child_chunk_ids = child_chunk_ids
        
        # Update other document details
        if title is not None:
            document_details.title = title
        if metadata is not None:
            document_details.metadata = metadata
        document_details.modification_timestamp=datetime.now(timezone.utc)
        
        # Save document details
        self.__document_details_repository.upsert(shard_name, document_details)