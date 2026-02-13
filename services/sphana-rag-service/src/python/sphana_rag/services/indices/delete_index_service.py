from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.constants import GLOBAL_SHARD_NAME
from sphana_rag.models import IndexDetails
from sphana_rag.repositories import IndexDetailsRepository, ShardDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChildChunkDetailsRepository, ParentChunkDetailsRepository
from sphana_rag.services.cluster import ClusterRouterService
from sphana_rag.utils import ShardUtil

TOPIC_DELETE_INDEX = "shard.delete_index"

@singleton
class DeleteIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository,
                 shard_details_repository: ShardDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 child_chunk_details_repository: ChildChunkDetailsRepository,
                 parent_chunk_details_repository: ParentChunkDetailsRepository,
                 cluster_router_service: ClusterRouterService):
        self.__index_details_repository = index_details_repository
        self.__shard_details_repository = shard_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository
        self.__cluster_router_service = cluster_router_service

        # Register listener for index write operations
        self.__cluster_router_service.listen(TOPIC_DELETE_INDEX, self._handle_delete_writes)

    def delete_index(self, index_name: str) -> None:
        # Route all write operations to the GLOBAL_SHARD_NAME owner
        message: dict = {
            "index_name": index_name
        }
        self.__cluster_router_service.route(GLOBAL_SHARD_NAME, TOPIC_DELETE_INDEX, message)
    
    def _handle_delete_writes(self, global_shard_name: str, message: dict) -> Optional[dict]:
        # Get message payload
        index_name: str = message["index_name"]

        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")

        for shard_number in range(index_details.number_of_shards):
            # Get shard name
            shard_name: str = ShardUtil.get_shard_name(index_name, shard_number)

            # Drop parent chunk details table
            self.__parent_chunk_details_repository.drop_table(shard_name)

            # Drop child chunk details table
            self.__child_chunk_details_repository.drop_table(shard_name)

            # Drop document details table
            self.__document_details_repository.drop_table(shard_name)

            # Drop index vectors index
            self.__index_vectors_repository.drop_index(shard_name)

            # Delete shard details
            self.__shard_details_repository.delete(shard_name)

        # Delete index details
        self.__index_details_repository.delete(index_name)
        return None