import sys
from injector import inject, singleton
from sphana_data_store.constants import GLOBAL_SHARD_NAME
from sphana_data_store.configs import Data StoreConfig
from sphana_data_store.models import ListResults, ShardDetails
from sphana_data_store.repositories import ShardDetailsRepository

@singleton
class ClusterNodesService:
    
    @inject
    def __init__(self, 
                 store_config: Data StoreConfig,
                 shard_details_repository: ShardDetailsRepository):
        self.store_config = store_config
        self.shard_details_repository = shard_details_repository

    def get_node(self, shard_name: str) -> str:
        # Assert nodes
        if len(self.store_config.nodes) == 0:
            raise ValueError("No nodes configured in Data Store config")
        
        # Check if global shard
        if shard_name is GLOBAL_SHARD_NAME:
            return self.store_config.nodes[0]

        # Compute shard index
        all_shards: ListResults[ShardDetails] = self.shard_details_repository.list(None, sys.maxsize)
        shard_index: int = -1
        for i, shard_details in enumerate(all_shards.documents):
            if shard_details.shard_name == shard_name:
                shard_index = i
                break
        if shard_index == -1:
            raise ValueError(f"Shard with name {shard_name} not found")
        
        # Compute node index and return node
        node_index: int = shard_index % len(self.store_config.nodes)
        return self.store_config.nodes[node_index]