from datetime import datetime, timezone
from injector import inject, singleton
from managed_exceptions import ItemAlreadyExistsException
from sphana_data_store.models import IndexDetails, ShardDetails
from sphana_data_store.repositories import IndexDetailsRepository, ShardDetailsRepository, IndexVectorsRepository, EntryDetailsRepository, ChunkDetailsRepository, EmbeddingDetailsRepository
from sphana_data_store.utils import ShardUtil

@singleton
class CreateIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository,
                 shard_details_repository: ShardDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 entry_details_repository: EntryDetailsRepository,
                 chunk_details_repository: ChunkDetailsRepository,
                 embedding_details_repository: EmbeddingDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__shard_details_repository = shard_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__entry_details_repository = entry_details_repository
        self.__chunk_details_repository = chunk_details_repository
        self.__embedding_details_repository = embedding_details_repository

    def create_index(self, index_name: str, description: str, number_of_shards: int, max_parent_chunk_size: int, max_child_chunk_size: int, parent_chunk_overlap_size: int, child_chunk_overlap_size: int) -> None:
        # Assert index name
        if self.__index_details_repository.exists(index_name):
            raise ItemAlreadyExistsException(f"Index {index_name} already exists")

        for shard_number in range(number_of_shards):
            # Get shard name
            shard_name: str = ShardUtil.get_shard_name(index_name, shard_number)

            # Init embedding details table
            self.__embedding_details_repository.init_table(shard_name)

            # Init chunk details table
            self.__chunk_details_repository.init_table(shard_name)

            # Init entry details table
            self.__entry_details_repository.init_table(shard_name)

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