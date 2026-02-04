from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, DocumentDetails
from sphana_rag.repositories import IndexDetailsRepository, DocumentDetailsRepository

@singleton
class GetDocumentService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 document_details_repository: DocumentDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__document_details_repository = document_details_repository

    def get_document(self, index_name: str, document_id: str) -> DocumentDetails:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Get document details
        document_details: Optional[DocumentDetails] = self.__document_details_repository.read(index_name, document_id)
        if document_details is None:
            raise ItemNotFoundException(f"Document {document_id} does not exist in index {index_name}")
        
        return document_details
