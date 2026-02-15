from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_store.models import IndexDetails, ChunkDetails, EmbeddingDetails, TextChunkResult, ExecuteQueryResult
from sphana_store.repositories import IndexDetailsRepository, IndexVectorsRepository, ChunkDetailsRepository, EmbeddingDetailsRepository
from sphana_store.utils import ShardUtil

class SearchChunkResult:
    def __init__(self, shard_name: str, embedding_id: str, score: float):
        self.shard_name = shard_name
        self.embedding_id = embedding_id
        self.score = score

@singleton
class ExecuteQueryService:

    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 chunk_details_repository: ChunkDetailsRepository,
                 embedding_details_repository: EmbeddingDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__chunk_details_repository = chunk_details_repository
        self.__embedding_details_repository = embedding_details_repository

    # TODO: make sure to add f"search_query: {query}" prefix to the query!
    def execute_query(self, index_name: str, query_embedding: list[float], max_results: int, score_threshold: Optional[float] = None) -> list[ExecuteQueryResult]:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Search for similar child chunks across all shards
        total_search_results: list[SearchChunkResult] = []
        for shard_number in range(index_details.number_of_shards):
            shard_name: str = ShardUtil.get_shard_name(index_name, shard_number)
            search_results: list[TextChunkResult] = self.__index_vectors_repository.search(shard_name, query_embedding, max_results)
            for search_result in search_results:
                total_search_results.append(SearchChunkResult(shard_name, search_result.embedding_id, search_result.score))

        # Sort results by score and limit
        total_search_results.sort(key=lambda x: x.score, reverse=False)

        # Deduplicate by parent chunk: for each child hit, look up its parent.
        # Keep the best (first) score per unique parent chunk.
        seen_chunk_ids: set[str] = set()
        results: list[ExecuteQueryResult] = []
        
        for search_result in total_search_results:
            if len(results) >= max_results:
                break
            
            # Filter by score threshold (L2 distance: lower = better)
            if score_threshold is not None and search_result.score > score_threshold:
                break  # Results are sorted ascending, so all remaining are worse
                
            # Look up embedding details to find its chunk
            embedding_details: Optional[EmbeddingDetails] = self.__embedding_details_repository.read(search_result.shard_name, search_result.embedding_id)
            if embedding_details is None:
                # TODO: This should not happen, log warning
                continue
            
            # Deduplicate by chunk id
            if embedding_details.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(embedding_details.chunk_id)
            
            # Look up chunk details to get content
            chunk_details: Optional[ChunkDetails] = self.__chunk_details_repository.read(search_result.shard_name, embedding_details.chunk_id)
            if chunk_details is None:
                # TODO: This should not happen, log warning
                continue
            
            actual_result: ExecuteQueryResult = ExecuteQueryResult(
                entry_id=chunk_details.entry_id,
                chunk_id=chunk_details.chunk_id,
                payload=chunk_details.payload,
                score=search_result.score
            )
            results.append(actual_result)
        
        return results