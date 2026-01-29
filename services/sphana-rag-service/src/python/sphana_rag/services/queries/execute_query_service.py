from typing import Optional
from fastapi import Depends
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, ChunkDetails, TextChunkDetails, TextChunkResult, ExecuteQueryResult
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, ChunkDetailsRepository
from sphana_rag.services.tokenizer import TextTokenizer

class ExecuteQueryService:

    def __init__(self,
                 index_details_repository: IndexDetailsRepository = Depends(IndexDetailsRepository),
                 index_vectors_repository: IndexVectorsRepository = Depends(IndexVectorsRepository),
                 chunk_details_repository: ChunkDetailsRepository = Depends(ChunkDetailsRepository),
                 text_tokenizer: TextTokenizer = Depends(TextTokenizer)):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__chunk_details_repository = chunk_details_repository
        self.__text_tokenizer = text_tokenizer

    def execute_query(self, index_name: str, query: str, max_results: int) -> list[ExecuteQueryResult]:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details == None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Generate embedding for the query
        query_embedding_details: list[TextChunkDetails] = self.__text_tokenizer.chunk_text(query, 99999999, 0)
        query_embedding: list[float] = query_embedding_details[0].embedding

        # Search for similar chunks
        search_results: list[TextChunkResult] = self.__index_vectors_repository.search(index_name, query_embedding, max_results)

        # Retrieve chunks based on chunk IDs, and build results
        results: list[ExecuteQueryResult] = []
        for search_result in search_results:
            chunk_details: Optional[ChunkDetails] = self.__chunk_details_repository.read(index_name, search_result.chunk_id)
            if chunk_details == None:
                # TODO: This should not happen, log warning
                continue
            actual_result: ExecuteQueryResult = ExecuteQueryResult(
                chunk_id=chunk_details.chunk_id,
                document_id=chunk_details.document_id,
                content=chunk_details.content,
                score=search_result.score
            )
            results.append(actual_result)
        return results