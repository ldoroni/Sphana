from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_data_store.models import IndexDetails
from sphana_data_store.repositories import IndexDetailsRepository

@singleton
class GetIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository):
        self.__index_details_repository = index_details_repository

    def get_index(self, index_name: str) -> IndexDetails:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")

        return index_details