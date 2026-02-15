from datetime import datetime, timezone
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_store.models import IndexDetails, EntryDetails, ChunkDetails, EmbeddingDetails
from sphana_store.repositories import IndexDetailsRepository, IndexVectorsRepository, EntryDetailsRepository, ChunkDetailsRepository, EmbeddingDetailsRepository
from sphana_store.utils import ShardUtil

@singleton
class IngestChunkService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 entry_details_repository: EntryDetailsRepository,
                 chunk_details_repository: ChunkDetailsRepository,
                 embedding_details_repository: EmbeddingDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__entry_details_repository = entry_details_repository
        self.__chunk_details_repository = chunk_details_repository
        self.__embedding_details_repository = embedding_details_repository

    def ingest_chunk(self, index_name: str, entry_id: str, payload: bytes, embeddings: list[list[float]]):
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
        
        # Get chunk id
        chunk_id: str = self.__chunk_details_repository.new_chunk_id(shard_name, entry_id)

        # Save embeddings
        embedding_ids: list[str] = []
        for embedding in embeddings:
            embedding_id: str = self.__embedding_details_repository.next_unique_id(shard_name)
            embedding_details: EmbeddingDetails = EmbeddingDetails(
                embedding_id=embedding_id,
                chunk_id=chunk_id,
                creation_timestamp=datetime.now(timezone.utc)
                # TODO: save also embedding vector?
            )
            self.__embedding_details_repository.upsert(shard_name, embedding_details)
            self.__index_vectors_repository.ingest(shard_name, embedding_id, embedding)
            embedding_ids.append(embedding_id)

        # Save chunk details
        chunk_details: ChunkDetails = ChunkDetails(
            chunk_id=chunk_id,
            entry_id=entry_id,
            payload=payload,
            embedding_ids=embedding_ids,
            creation_timestamp=datetime.now(timezone.utc)
        )
        self.__chunk_details_repository.upsert(shard_name, chunk_details)