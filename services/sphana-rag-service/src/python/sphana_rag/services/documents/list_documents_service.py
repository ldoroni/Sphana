from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, DocumentDetails, ListResults
from sphana_rag.repositories import IndexDetailsRepository, DocumentDetailsRepository

@singleton
class ListDocumentsService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 document_details_repository: DocumentDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__document_details_repository = document_details_repository

    def list_documents(self, index_name: str,  offset: Optional[str], limit: int) -> ListResults[DocumentDetails]:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # List documents details
        return self.__document_details_repository.list(index_name, offset, limit)
