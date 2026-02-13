from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.constants import GLOBAL_SHARD_NAME
from sphana_rag.models import IndexDetails
from sphana_rag.repositories import IndexDetailsRepository
from sphana_rag.services.cluster import ClusterRouterService

TOPIC_UPDATE_INDEX = "shard.update_index"

@singleton
class UpdateIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository,
                 cluster_router_service: ClusterRouterService):
        self.__index_details_repository = index_details_repository
        self.__cluster_router_service = cluster_router_service

        # Register listener for index write operations
        self.__cluster_router_service.listen(TOPIC_UPDATE_INDEX, self._handle_update_writes)

    def update_index(self, index_name: str, description: Optional[str]) -> None:
         # Route all write operations to the GLOBAL_SHARD_NAME owner
        message: dict = {
            "index_name": index_name,
            "description": description
        }
        self.__cluster_router_service.route(GLOBAL_SHARD_NAME, TOPIC_UPDATE_INDEX, message)

    def _handle_update_writes(self, global_shard_name: str, message: dict) -> Optional[dict]:
        # Get message payload
        index_name: str = message["index_name"]
        description: Optional[str] = message["description"]
        
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")

        # Save index details
        if description is not None:
            index_details.description = description
        index_details.modification_timestamp = datetime.now(timezone.utc)
        self.__index_details_repository.upsert(index_details)
        return None