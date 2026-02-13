from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemAlreadyExistsException
from sphana_rag.constants import GLOBAL_SHARD_NAME
from sphana_rag.models import IndexDetails, ShardDetails
from sphana_rag.repositories import IndexDetailsRepository, ShardDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChildChunkDetailsRepository, ParentChunkDetailsRepository
from sphana_rag.services.cluster import ClusterRouterService
from sphana_rag.utils import ShardUtil

TOPIC_CREATE_INDEX = "shard.create_index"

@singleton
class CreateIndexService:
    
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
        self.__cluster_router_service.listen(TOPIC_CREATE_INDEX, self._handle_create_writes)

    def create_index(self, index_name: str, description: str, number_of_shards: int, max_parent_chunk_size: int, max_child_chunk_size: int, parent_chunk_overlap_size: int, child_chunk_overlap_size: int) -> None:
        # Route all write operations to the GLOBAL_SHARD_NAME owner
        message: dict = {
            "index_name": index_name,
            "description": description,
            "number_of_shards": number_of_shards,
            "max_parent_chunk_size": max_parent_chunk_size,
            "max_child_chunk_size": max_child_chunk_size,
            "parent_chunk_overlap_size": parent_chunk_overlap_size,
            "child_chunk_overlap_size": child_chunk_overlap_size
        }
        self.__cluster_router_service.route(GLOBAL_SHARD_NAME, TOPIC_CREATE_INDEX, message)
        
    def _handle_create_writes(self, global_shard_name: str, message: dict) -> Optional[dict]:
        # Get message payload
        index_name: str = message["index_name"]
        description: str = message["description"]
        number_of_shards: int = message["number_of_shards"]
        max_parent_chunk_size: int = message["max_parent_chunk_size"]
        max_child_chunk_size: int = message["max_child_chunk_size"]
        parent_chunk_overlap_size: int = message["parent_chunk_overlap_size"]
        child_chunk_overlap_size: int = message["child_chunk_overlap_size"]

        # Assert index name
        if self.__index_details_repository.exists(index_name):
            raise ItemAlreadyExistsException(f"Index {index_name} already exists")
        
        # Init shards
        for shard_number in range(number_of_shards):
            # Get shard name
            shard_name: str = ShardUtil.get_shard_name(index_name, shard_number)

            # Init parent chunk details table
            self.__parent_chunk_details_repository.init_table(shard_name)

            # Init child chunk details table
            self.__child_chunk_details_repository.init_table(shard_name)

            # Init document details table
            self.__document_details_repository.init_table(shard_name)

            # Init index vectors index
            self.__index_vectors_repository.init_index(shard_name)

            # Save shard details
            shard_details: ShardDetails = ShardDetails(
                shard_name=shard_name,
                index_name=index_name,
                creation_timestamp=datetime.now(timezone.utc),
                modification_timestamp=datetime.now(timezone.utc)
            )
            self.__shard_details_repository.upsert(shard_details)

        # Save index details
        index_details: IndexDetails = IndexDetails(
            index_name=index_name,
            description=description,
            number_of_shards=number_of_shards,
            max_parent_chunk_size=max_parent_chunk_size,
            max_child_chunk_size=max_child_chunk_size,
            parent_chunk_overlap_size=parent_chunk_overlap_size,
            child_chunk_overlap_size=child_chunk_overlap_size,
            creation_timestamp=datetime.now(timezone.utc),
            modification_timestamp=datetime.now(timezone.utc)
        )
        self.__index_details_repository.upsert(index_details)
        return None