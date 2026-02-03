from typing import Optional
from injector import inject, singleton
from sphana_rag.models import IndexDetails
from sphana_rag.models import ListResults
from sphana_rag.repositories import IndexDetailsRepository

@singleton
class ListIndicesService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository):
        self.__index_details_repository = index_details_repository

    def list_indices(self, offset: Optional[str], limit: int) -> ListResults[IndexDetails]:
        return self.__index_details_repository.list(offset, limit)