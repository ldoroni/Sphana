from injector import inject, singleton
from sphana_rag.controllers.queries.v1.schemas import ExecuteQueryRequest, ExecuteQueryResponse, ExecuteQueryResult
from sphana_rag.services.queries import ExecuteQueryService
from sphana_rag.utils import CompressionUtil
from request_handler import RequestHandler

@singleton
class ExecuteQueryHandler(RequestHandler[ExecuteQueryRequest, ExecuteQueryResponse]):

    @inject
    def __init__(self, 
                 execute_query_service: ExecuteQueryService):
        super().__init__()
        self.__execute_query_service = execute_query_service

    def _on_validate(self, request: ExecuteQueryRequest):
        # Validate request
        pass

    def _on_invoke(self, request: ExecuteQueryRequest) -> ExecuteQueryResponse:
        # Execute query
        results = self.__execute_query_service.execute_query(
            index_name=request.index_name or "",
            query=request.query or "",
            max_results=request.max_results or 0
        )

        # Return response
        return ExecuteQueryResponse(
            results=[
                ExecuteQueryResult(
                    document_id=result.document_id,
                    chunk_index=result.chunk_index,
                    content=CompressionUtil.decompress(result.content), # TODO: I dislike the decompression here!
                    score=result.score
                )
                for result in results
            ]
        )
