from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_data_store.models import IndexDetails
from sphana_data_store.repositories import IndexDetailsRepository, ShardDetailsRepository, IndexVectorsRepository, EntryDetailsRepository, EntryPayloadsRepository, EmbeddingDetailsRepository
from sphana_data_store.utils import ShardUtil

@singleton
class DeleteIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository,
                 shard_details_repository: ShardDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 entry_details_repository: EntryDetailsRepository,
                 entry_payloads_repository: EntryPayloadsRepository,
                 embedding_details_repository: EmbeddingDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__shard_details_repository = shard_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__entry_details_repository = entry_details_repository
        self.__entry_payloads_repository = entry_payloads_repository
        self.__embedding_details_repository = embedding_details_repository

    def delete_index(self, index_name: str) -> None:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")

        for shard_number in range(index_details.number_of_shards):
            # Get shard name
            shard_name: str = ShardUtil.get_shard_name(index_name, shard_number)

            # Drop embedding details table
            self.__embedding_details_repository.drop_table(shard_name)

            # Drop entry payloads storage
            self.__entry_payloads_repository.drop_storage(shard_name)

            # Drop entry details table
            self.__entry_details_repository.drop_table(shard_name)

            # Drop index vectors index
            self.__index_vectors_repository.drop_index(shard_name)

            # Delete shard details
            self.__shard_details_repository.delete(shard_name)

        # Delete index details
        self.__index_details_repository.delete(index_name)