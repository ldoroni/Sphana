from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, DocumentDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChunkDetailsRepository

@singleton
class DeleteDocumentService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 chunk_details_repository: ChunkDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__chunk_details_repository = chunk_details_repository

    def delete_document(self, index_name: str, document_id: str):
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Get document details
        document_details: Optional[DocumentDetails] = self.__document_details_repository.read(index_name, document_id)
        if document_details is None:
            raise ItemNotFoundException(f"Document {document_id} does not exist in index {index_name}")
        
        # Delete chunks details
        for chunk_id in document_details.chunk_ids:
            self.__chunk_details_repository.delete(index_name, chunk_id)
            self.__index_vectors_repository.delete(index_name, chunk_id)

        # Delete document details
        self.__document_details_repository.delete(index_name, document_id)
