from injector import inject, singleton
from sphana_store.controllers.queries.v1.schemas import ExecuteQueryRequest, ExecuteQueryResponse, ExecuteQueryResult
from sphana_store.services.queries import ExecuteQueryService
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
            query_embedding=request.query_embedding or [],
            max_results=request.max_results or 0,
            score_threshold=request.score_threshold
        )

        # Return response
        return ExecuteQueryResponse(
            results=[
                ExecuteQueryResult(
                    entry_id=result.entry_id,
                    chunk_id=result.chunk_id,
                    payload=result.payload,
                    score=result.score
                )
                for result in results
            ]
        )
