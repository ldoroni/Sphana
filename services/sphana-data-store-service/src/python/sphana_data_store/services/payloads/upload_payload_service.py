from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_data_store.models import IndexDetails, EntryDetails
from sphana_data_store.repositories import IndexDetailsRepository, EntryDetailsRepository, EntryPayloadsRepository
from sphana_data_store.utils import ShardUtil

@singleton
class UploadPayloadService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 entry_details_repository: EntryDetailsRepository,
                 entry_payloads_repository: EntryPayloadsRepository):
        self.__index_details_repository = index_details_repository
        self.__entry_details_repository = entry_details_repository
        self.__entry_payloads_repository = entry_payloads_repository

    def upload_payload(self, index_name: str, entry_id: str, payload: bytes):
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
        
        # Save entry payload
        self.__entry_payloads_repository.save(shard_name, entry_id, payload)