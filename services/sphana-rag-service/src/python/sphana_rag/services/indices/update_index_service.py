from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.constants import GLOBAL_SHARD_NAME
from sphana_rag.models import IndexDetails
from sphana_rag.repositories import IndexDetailsRepository, DistributedCacheRepository

@singleton
class UpdateIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository,
                 distributed_cache_repository: DistributedCacheRepository):
        self.__index_details_repository = index_details_repository
        self.__distributed_cache_repository = distributed_cache_repository

    def update_index(self, index_name: str, description: Optional[str]) -> None:
        # Save information in the repository
        with self.__distributed_cache_repository.lock(GLOBAL_SHARD_NAME, ttl_seconds=300):
            # Get index details
            index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
            if index_details is None:
                raise ItemNotFoundException(f"Index {index_name} does not exist")

            # Save index details
            if description is not None:
                index_details.description = description
            index_details.modification_timestamp = datetime.now(timezone.utc)
            self.__index_details_repository.upsert(index_details)