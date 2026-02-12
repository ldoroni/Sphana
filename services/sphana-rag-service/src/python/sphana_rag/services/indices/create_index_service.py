from datetime import datetime, timezone
from injector import inject, singleton
from managed_exceptions import ItemAlreadyExistsException
from sphana_rag.models import IndexDetails, ShardDetails
from sphana_rag.repositories import IndexDetailsRepository, ShardDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChildChunkDetailsRepository, ParentChunkDetailsRepository
from sphana_rag.utils import ShardUtil

@singleton
class CreateIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository,
                 shard_details_repository: ShardDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 child_chunk_details_repository: ChildChunkDetailsRepository,
                 parent_chunk_details_repository: ParentChunkDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__shard_details_repository = shard_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository

    def create_index(self, index_name: str, description: str, number_of_shards: int, max_parent_chunk_size: int, max_child_chunk_size: int, parent_chunk_overlap_size: int, child_chunk_overlap_size: int) -> None:
        # Assert index name
        if self.__index_details_repository.exists(index_name):
            raise ItemAlreadyExistsException(f"Index {index_name} already exists")

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