from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException, ItemAlreadyExistsException
from sphana_store.models import IndexDetails, EntryDetails
from sphana_store.repositories import IndexDetailsRepository, EntryDetailsRepository
from sphana_store.utils import ShardUtil

@singleton
class CreateEntryService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 entry_details_repository: EntryDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__entry_details_repository = entry_details_repository

    def create_entry(self, index_name: str, entry_id: str, title: str, metadata: dict[str, str]) -> None:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Get shard name
        shard_name: str = ShardUtil.compute_shard_name(
            index_name, 
            entry_id, 
            index_details.number_of_shards
        )
        
        # Assert entry does not already exist
        if self.__entry_details_repository.exists(shard_name, entry_id):
            raise ItemAlreadyExistsException(f"Entry {entry_id} already exists in index {index_name}")
        
        #  Save entry details
        entry_details: EntryDetails = EntryDetails(
            entry_id=entry_id,
            title=title,
            metadata=metadata,
            creation_timestamp=datetime.now(timezone.utc),
            modification_timestamp=datetime.now(timezone.utc)
        )
        self.__entry_details_repository.upsert(shard_name, entry_details)