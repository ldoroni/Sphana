from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_store.models import IndexDetails
from sphana_store.repositories import IndexDetailsRepository

@singleton
class UpdateIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository):
        self.__index_details_repository = index_details_repository

    def update_index(self, index_name: str, description: Optional[str]) -> None:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")

        # Save index details
        if description is not None:
            index_details.description = description
        index_details.modification_timestamp = datetime.now(timezone.utc)
        self.__index_details_repository.upsert(index_details)