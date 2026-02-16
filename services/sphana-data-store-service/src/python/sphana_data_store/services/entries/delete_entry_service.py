from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_data_store.models import IndexDetails, EntryDetails
from sphana_data_store.services.chunks import ClearChunksService
from sphana_data_store.repositories import IndexDetailsRepository, EntryDetailsRepository
from sphana_data_store.utils import ShardUtil

@singleton
class DeleteEntryService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 entry_details_repository: EntryDetailsRepository,
                 clear_chunks_service: ClearChunksService):
        self.__index_details_repository = index_details_repository
        self.__entry_details_repository = entry_details_repository
        self.__clear_chunks_service = clear_chunks_service

    def delete_entry(self, index_name: str, entry_id: str):
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
        
        # Delete entry chunks
        self.__clear_chunks_service.clear_chunks(index_name, entry_id)

        # Delete entry details
        self.__entry_details_repository.delete(shard_name, entry_id)