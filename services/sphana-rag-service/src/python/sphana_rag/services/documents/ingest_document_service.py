from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException, ItemAlreadyExistsException
from sphana_rag.models import IndexDetails, DocumentDetails, ChunkDetails, TextChunkDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChunkDetailsRepository
from sphana_rag.services.tokenizer import TextTokenizer
from sphana_rag.services.utils import ShardUtil

@singleton
class IngestDocumentService:
    
    @inject
    def __init__(self,
                 shard_util: ShardUtil,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 chunk_details_repository: ChunkDetailsRepository,
                 text_tokenizer: TextTokenizer):
        self.__shard_util = shard_util
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__chunk_details_repository = chunk_details_repository
        self.__text_tokenizer = text_tokenizer

    def ingest_document(self, index_name: str, document_id: str, title: str, content: str, metadata: dict[str, str]):
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Get shard name
        shard_name: str = self.__shard_util.compute_shard_name(
            index_name, 
            document_id, 
            index_details.number_of_shards
        )
        
        # Assert document id
        if self.__document_details_repository.exists(shard_name, document_id):
            raise ItemAlreadyExistsException(f"Document {document_id} already exists in index {index_name}")
        
        # Chunk document content
        chunks: list[TextChunkDetails] = self.__text_tokenizer.tokenize_and_chunk_text(
            content, 
            max_chunk_size=index_details.max_chunk_size, 
            chunk_overlap_size=index_details.chunk_overlap_size
        )

        # Save chunks details
        chunk_ids: list[str] = []
        for chunk_index in range(len(chunks)):
            chunk: TextChunkDetails = chunks[chunk_index]
            chunk_id: str = self.__chunk_details_repository.next_unique_id(shard_name)
            chunk_details: ChunkDetails = ChunkDetails(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                content=chunk.text
            )
            self.__chunk_details_repository.upsert(shard_name, chunk_details)
            self.__index_vectors_repository.ingest(shard_name, chunk_id, chunk.embedding)
            chunk_ids.append(chunk_id)
        
        # Save document details
        document_details: DocumentDetails = DocumentDetails(
            document_id=document_id,
            title=title,
            content=content,
            chunk_ids=chunk_ids,
            metadata=metadata,
            creation_timestamp=datetime.now(timezone.utc),
            modification_timestamp=datetime.now(timezone.utc)
        )
        self.__document_details_repository.upsert(shard_name, document_details)
