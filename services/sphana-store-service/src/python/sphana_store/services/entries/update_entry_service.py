from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_store.models import IndexDetails, EntryDetails
from sphana_store.repositories import IndexDetailsRepository, EntryDetailsRepository
from sphana_store.utils import ShardUtil

@singleton
class UpdateEntryService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 entry_details_repository: EntryDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__entry_details_repository = entry_details_repository

    def update_entry(self, index_name: str, entry_id: str, title: Optional[str], metadata: Optional[dict[str, str]]):
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
        
        # Get entry details
        entry_details: Optional[EntryDetails] = self.__entry_details_repository.read(shard_name, entry_id)
        if entry_details is None:
            raise ItemNotFoundException(f"Entry {entry_id} does not exist in index {index_name}")
        
        # Update entry details
        if title is not None:
            entry_details.title = title

        if metadata is not None:
            entry_details.metadata = metadata
        
        entry_details.modification_timestamp=datetime.now(timezone.utc)
        
        # Save entry details
        self.__entry_details_repository.upsert(shard_name, entry_details)