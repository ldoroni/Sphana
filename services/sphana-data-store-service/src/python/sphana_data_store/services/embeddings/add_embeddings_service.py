from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_data_store.models import IndexDetails, EntryDetails, EmbeddingDetails
from sphana_data_store.repositories import IndexDetailsRepository, IndexVectorsRepository, EntryDetailsRepository, EmbeddingDetailsRepository
from sphana_data_store.utils import ShardUtil

@singleton
class AddEmbeddingsService:
    
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

    def add_embeddings(self, index_name: str, entry_id: str, start_indexes: list[int], end_indexes: list[int], embeddings: list[list[float]]):
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
        
        # Save embeddings
        embedding_ids: list[str] = []
        for embedding, start_index, end_index in zip(embeddings, start_indexes, end_indexes):
            embedding_id: str = self.__embedding_details_repository.new_embedding_id(shard_name, entry_id)
            embedding_details: EmbeddingDetails = EmbeddingDetails(
                embedding_id=embedding_id,
                entry_id=entry_id,
                start_index=start_index,
                end_index=end_index,
                creation_timestamp=datetime.now(timezone.utc)
            )
            self.__embedding_details_repository.upsert(shard_name, embedding_details)
            self.__index_vectors_repository.add(shard_name, embedding_id, embedding)
            embedding_ids.append(embedding_id)
