from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, ChildChunkDetails, ParentChunkDetails, TextChunkResult, ExecuteQueryResult
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, ChildChunkDetailsRepository, ParentChunkDetailsRepository
from sphana_rag.services.tokenizer import TextEmbedderService
from sphana_rag.utils import ShardUtil

class SearchChunkResult:
    def __init__(self, shard_name: str, chunk_id: str, score: float):
        self.shard_name = shard_name
        self.chunk_id = chunk_id
        self.score = score

@singleton
class ExecuteQueryService:

    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 child_chunk_details_repository: ChildChunkDetailsRepository,
                 parent_chunk_details_repository: ParentChunkDetailsRepository,
                 text_embedder_service: TextEmbedderService):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository
        self.__text_embedder_service = text_embedder_service

    def execute_query(self, index_name: str, query: str, max_results: int, score_threshold: Optional[float] = None) -> list[ExecuteQueryResult]:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Generate embedding for the query
        # Using "search_query" prefix for query embeddings as per nomic best practices
        prefixed_query = f"search_query: {query}"
        query_embedding: list[float] = self.__text_embedder_service.embed_text(prefixed_query)

        # Search for similar child chunks across all shards
        total_search_results: list[SearchChunkResult] = []
        for shard_number in range(index_details.number_of_shards):
            shard_name: str = ShardUtil.get_shard_name(index_name, shard_number)
            search_results: list[TextChunkResult] = self.__index_vectors_repository.search(shard_name, query_embedding, max_results)
            for search_result in search_results:
                total_search_results.append(SearchChunkResult(shard_name, search_result.chunk_id, search_result.score))

        # Sort results by score and limit
        total_search_results.sort(key=lambda x: x.score, reverse=False)

        # Deduplicate by parent chunk: for each child hit, look up its parent.
        # Keep the best (first) score per unique parent chunk.
        seen_parent_ids: set[str] = set()
        results: list[ExecuteQueryResult] = []
        
        for search_result in total_search_results:
            if len(results) >= max_results:
                break
            
            # Filter by score threshold (L2 distance: lower = better)
            if score_threshold is not None and search_result.score > score_threshold:
                break  # Results are sorted ascending, so all remaining are worse
                
            # Look up child chunk to find its parent
            child_chunk: Optional[ChildChunkDetails] = self.__child_chunk_details_repository.read(search_result.shard_name, search_result.chunk_id)
            if child_chunk is None:
                # TODO: This should not happen, log warning
                continue
            
            # Deduplicate by parent chunk id
            if child_chunk.parent_chunk_id in seen_parent_ids:
                continue
            seen_parent_ids.add(child_chunk.parent_chunk_id)
            
            # Look up parent chunk to get content
            parent_chunk: Optional[ParentChunkDetails] = self.__parent_chunk_details_repository.read(search_result.shard_name, child_chunk.parent_chunk_id)
            if parent_chunk is None:
                # TODO: This should not happen, log warning
                continue
            
            actual_result: ExecuteQueryResult = ExecuteQueryResult(
                document_id=parent_chunk.document_id,
                chunk_index=parent_chunk.chunk_index,
                content=parent_chunk.content,
                score=search_result.score
            )
            results.append(actual_result)
        
        return results