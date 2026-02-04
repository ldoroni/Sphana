from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, DocumentDetails, ChunkDetails, TextChunkDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChunkDetailsRepository
from sphana_rag.services.tokenizer import TextTokenizer

@singleton
class UpdateDocumentService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 chunk_details_repository: ChunkDetailsRepository,
                 text_tokenizer: TextTokenizer):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__chunk_details_repository = chunk_details_repository
        self.__text_tokenizer = text_tokenizer

    def update_document(self, index_name: str, document_id: str, title: Optional[str], content: Optional[str], metadata: Optional[dict[str, str]]):
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Get document details
        document_details: Optional[DocumentDetails] = self.__document_details_repository.read(index_name, document_id)
        if document_details is None:
            raise ItemNotFoundException(f"Document {document_id} does not exist in index {index_name}")
        
        if content is not None:
            # Chunk new document content
            chunks: list[TextChunkDetails] = self.__text_tokenizer.chunk_text(
                content, 
                max_chunk_size=index_details.max_chunk_size, 
                max_chunk_overlap_size=index_details.max_chunk_overlap_size
            )

            # Delete old chunks details
            for chunk_id in document_details.chunk_ids:
                self.__chunk_details_repository.delete(index_name, chunk_id)
                self.__index_vectors_repository.delete(index_name, chunk_id)

            # Save new chunks details
            chunk_ids: list[str] = []
            for chunk_index in range(len(chunks)):
                chunk: TextChunkDetails = chunks[chunk_index]
                chunk_id: str = self.__chunk_details_repository.next_unique_id(index_name)
                chunk_details: ChunkDetails = ChunkDetails(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=chunk.text
                )
                self.__chunk_details_repository.upsert(index_name, chunk_details)
                self.__index_vectors_repository.ingest(index_name, chunk_id, chunk.embedding)
                chunk_ids.append(chunk_id)

            # Update document details
            document_details.content = content
            document_details.chunk_ids = chunk_ids
        
        # Update other document details
        if title is not None:
            document_details.title = title
        if metadata is not None:
            document_details.metadata = metadata
        document_details.modification_timestamp=datetime.now(timezone.utc)
        
        # Save document details
        self.__document_details_repository.upsert(index_name, document_details)
