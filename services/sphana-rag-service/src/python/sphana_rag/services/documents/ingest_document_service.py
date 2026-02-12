from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException, ItemAlreadyExistsException
from sphana_rag.models import IndexDetails, DocumentDetails, ParentChunkDetails, ChildChunkDetails, TextChunkDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ParentChunkDetailsRepository, ChildChunkDetailsRepository
from sphana_rag.services.tokenizer import TextTokenizer
from sphana_rag.utils import ShardUtil, CompressionUtil

@singleton
class IngestDocumentService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 parent_chunk_details_repository: ParentChunkDetailsRepository,
                 child_chunk_details_repository: ChildChunkDetailsRepository,
                 text_tokenizer: TextTokenizer):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__text_tokenizer = text_tokenizer

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
        
        # Chunk document content using parent-child chunking
        chunks: list[TextChunkDetails] = self.__text_tokenizer.tokenize_and_chunk_text(
            content, 
            max_parent_chunk_size=index_details.max_parent_chunk_size,
            max_child_chunk_size=index_details.max_child_chunk_size,
            parent_chunk_overlap_size=index_details.parent_chunk_overlap_size,
            child_chunk_overlap_size=index_details.child_chunk_overlap_size
        )

        # Group child chunks by parent_text to create parent-child relationships
        parent_text_to_children: dict[str, list[TextChunkDetails]] = {}
        parent_text_order: list[str] = []
        for chunk in chunks:
            if chunk.parent_text not in parent_text_to_children:
                parent_text_to_children[chunk.parent_text] = []
                parent_text_order.append(chunk.parent_text)
            parent_text_to_children[chunk.parent_text].append(chunk)

        # Save parent and child chunks
        parent_chunk_ids: list[str] = []
        child_chunk_ids: list[str] = []
        
        for parent_index, parent_text in enumerate(parent_text_order):
            # Create and save parent chunk
            parent_chunk_id: str = self.__parent_chunk_details_repository.next_unique_id(shard_name)
            parent_chunk_details: ParentChunkDetails = ParentChunkDetails(
                parent_chunk_id=parent_chunk_id,
                document_id=document_id,
                chunk_index=parent_index,
                content=CompressionUtil.compress(parent_text)
            )
            self.__parent_chunk_details_repository.upsert(shard_name, parent_chunk_details)
            parent_chunk_ids.append(parent_chunk_id)
            
            # Create and save child chunks for this parent
            children = parent_text_to_children[parent_text]
            for child_index, child_chunk in enumerate(children):
                child_chunk_id: str = self.__child_chunk_details_repository.next_unique_id(shard_name)
                child_chunk_details: ChildChunkDetails = ChildChunkDetails(
                    child_chunk_id=child_chunk_id,
                    parent_chunk_id=parent_chunk_id,
                    # document_id=document_id,
                    # child_chunk_index=child_index
                )
                self.__child_chunk_details_repository.upsert(shard_name, child_chunk_details)
                self.__index_vectors_repository.ingest(shard_name, child_chunk_id, child_chunk.embedding)
                child_chunk_ids.append(child_chunk_id)

        # Save document details
        document_details: DocumentDetails = DocumentDetails(
            document_id=document_id,
            title=title,
            content=CompressionUtil.compress(content),
            metadata=metadata,
            parent_chunk_ids=parent_chunk_ids,
            child_chunk_ids=child_chunk_ids,
            creation_timestamp=datetime.now(timezone.utc),
            modification_timestamp=datetime.now(timezone.utc)
        )
        self.__document_details_repository.upsert(shard_name, document_details)