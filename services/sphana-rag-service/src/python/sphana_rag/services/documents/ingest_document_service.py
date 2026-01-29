from datetime import datetime, timezone
from fastapi import Depends
from managed_exceptions import ItemNotFoundException, ItemAlreadyExistsException
from sphana_rag.models import DocumentDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChunkDetailsRepository

class IngestDocumentService:
    
    def __init__(self,
                 index_details_repository: IndexDetailsRepository = Depends(IndexDetailsRepository),
                 index_vectors_repository: IndexVectorsRepository = Depends(IndexVectorsRepository),
                 document_details_repository: DocumentDetailsRepository = Depends(DocumentDetailsRepository),
                 chunk_details_repository: ChunkDetailsRepository = Depends(ChunkDetailsRepository)):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__chunk_details_repository = chunk_details_repository

    def ingest_document(self, index_name: str, document_id: str, title: str, content: str, metadata: dict[str, str]):
        if not(self.__index_details_repository.exists(index_name)):
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        if self.__document_details_repository.exists(index_name, document_id):
            raise ItemAlreadyExistsException(f"Document {document_id} already exists in index {index_name}")
        
        # Save document details
        document_details: DocumentDetails = DocumentDetails(
            document_id=document_id,
            title=title,
            content=content,
            chunk_ids=[], #todo
            metadata=metadata,
            creation_timestamp=datetime.now(timezone.utc),
            modification_timestamp=datetime.now(timezone.utc)
        )
        self.__document_details_repository.upsert(index_name, document_details)
