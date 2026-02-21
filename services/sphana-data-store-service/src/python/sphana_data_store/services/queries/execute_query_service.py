from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_data_store.models import IndexDetails, EmbeddingDetails, EmbeddingResult, ExecuteQueryResult
from sphana_data_store.repositories import IndexDetailsRepository, IndexVectorsRepository, EmbeddingDetailsRepository, EntryPayloadsRepository
from sphana_data_store.utils import ShardUtil, Base64Util

class EmbeddingResultEx:
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
                 entry_payloads_repository: EntryPayloadsRepository,
                 embedding_details_repository: EmbeddingDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__entry_payloads_repository = entry_payloads_repository
        self.__embedding_details_repository = embedding_details_repository

    # TODO: make sure to add f"search_query: {query}" prefix to the query!
    def execute_query(self, index_name: str, query_embedding: list[float], max_results: int, score_threshold: Optional[float] = None) -> list[ExecuteQueryResult]:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Search for similar embeddings across all shards
        total_search_results: list[EmbeddingResultEx] = []
        for shard_number in range(index_details.number_of_shards):
            shard_name: str = ShardUtil.get_shard_name(index_name, shard_number)
            search_results: list[EmbeddingResult] = self.__index_vectors_repository.search(shard_name, query_embedding, max_results)
            for search_result in search_results:
                total_search_results.append(EmbeddingResultEx(shard_name, search_result.embedding_id, search_result.score))

        # Sort results by score and limit
        total_search_results.sort(key=lambda x: x.score, reverse=False)

        # Deduplicate by chunks.
        # Keep the best (first) score per unique chunk.
        # Key Format: shard_name:embedding_id:start_index:end_index
        seen_chunks: set[str] = set()
        results: list[ExecuteQueryResult] = []
        
        for search_result in total_search_results:
            if len(results) >= max_results:
                break
            
            # Filter by score threshold (L2 distance: lower = better)
            if score_threshold is not None and search_result.score > score_threshold:
                break  # Results are sorted ascending, so all remaining are worse
                
            # Look up embedding details to find its chunk
            embedding_details: Optional[EmbeddingDetails] = self.__embedding_details_repository.read(
                search_result.shard_name, 
                search_result.embedding_id
            )
            if embedding_details is None:
                # TODO: This should not happen, log warning
                continue
            
            # Deduplicate by chunk details
            compare_key: str =  search_result.shard_name + ":" + \
                                embedding_details.embedding_id + ":" + \
                                str(embedding_details.start_index) + ":" + \
                                str(embedding_details.end_index)
            if compare_key in seen_chunks:
                continue
            seen_chunks.add(compare_key)
            
            # Look up chunk payload
            payload: Optional[bytes] = self.__entry_payloads_repository.read_chunk(
                search_result.shard_name, 
                embedding_details.entry_id, 
                embedding_details.start_index,
                embedding_details.end_index
            )
            if payload is None:
                # TODO: This should not happen, log warning
                continue
            
            actual_result: ExecuteQueryResult = ExecuteQueryResult(
                entry_id=embedding_details.entry_id,
                payload=Base64Util.encode_to_bytes(payload),
                score=search_result.score
            )
            results.append(actual_result)
        
        return results