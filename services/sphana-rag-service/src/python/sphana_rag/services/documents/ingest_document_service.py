from datetime import datetime, timezone
from typing import Optional
from fastapi import Depends
from managed_exceptions import ItemNotFoundException, ItemAlreadyExistsException
from sphana_rag.models import IndexDetails, DocumentDetails, ChunkDetails, TextChunk
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChunkDetailsRepository
from sphana_rag.services.tokenizer import TextTokenizer

class IngestDocumentService:
    
    def __init__(self,
                 index_details_repository: IndexDetailsRepository = Depends(IndexDetailsRepository),
                 index_vectors_repository: IndexVectorsRepository = Depends(IndexVectorsRepository),
                 document_details_repository: DocumentDetailsRepository = Depends(DocumentDetailsRepository),
                 chunk_details_repository: ChunkDetailsRepository = Depends(ChunkDetailsRepository),
                 text_tokenizer: TextTokenizer = Depends(TextTokenizer)):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__chunk_details_repository = chunk_details_repository
        self.__text_tokenizer = text_tokenizer
        self.__id = 0

    def ingest_document(self, index_name: str, document_id: str, title: str, content: str, metadata: dict[str, str]):
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details == None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        if self.__document_details_repository.exists(index_name, document_id):
            raise ItemAlreadyExistsException(f"Document {document_id} already exists in index {index_name}")
        
        # Chunk document content
        chunks: list[TextChunk] = self.__text_tokenizer.chunk_text(
            content, 
            max_chunk_size=index_details.max_chunk_size, 
            max_chunk_overlap_size=index_details.max_chunk_overlap_size
        )

        # Save chunks details
        chunk_ids: list[str] = []
        for chunk_index in range(len(chunks)):
            chunk: TextChunk = chunks[chunk_index]
            chunk_id: str = str(self.__id) # TODO: Generate unique chunk ID
            self.__id+=1 # TODO: remove
            chunk_details: ChunkDetails = ChunkDetails(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                content=chunk.text
            )
            self.__chunk_details_repository.upsert(index_name, chunk_details)
            self.__index_vectors_repository.ingest(index_name, chunk_id, chunk.embedding)
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
        self.__document_details_repository.upsert(index_name, document_details)
