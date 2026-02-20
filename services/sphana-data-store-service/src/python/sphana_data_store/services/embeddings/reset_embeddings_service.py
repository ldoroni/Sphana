from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_data_store.models import IndexDetails, EntryDetails, EmbeddingDetails
from sphana_data_store.repositories import IndexDetailsRepository, IndexVectorsRepository, EntryDetailsRepository, EmbeddingDetailsRepository
from sphana_data_store.utils import ShardUtil

@singleton
class ResetEmbeddingsService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 entry_details_repository: EntryDetailsRepository,
                 embedding_details_repository: EmbeddingDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__entry_details_repository = entry_details_repository
        self.__embedding_details_repository = embedding_details_repository

    def reset_embeddings(self, index_name: str, entry_id: str):
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
        
        # List embeddings details
        embeddings_details: list[EmbeddingDetails] = self.__embedding_details_repository.list_for_entry(shard_name, entry_id)

        # Delete embeddings details and vectors
        for embedding_details in embeddings_details:
            self.__embedding_details_repository.delete(shard_name, embedding_details.embedding_id)
            self.__index_vectors_repository.delete(shard_name, embedding_details.embedding_id)

        # Delete entry details
        self.__entry_details_repository.delete(shard_name, entry_id)