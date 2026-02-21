from injector import inject, singleton
from request_handler import RequestHandler
from sphana_text_facade.controllers.documents.v1.schemas import QueryDocumentsRequest, QueryDocumentsResponse, QueryDocumentsResult
from sphana_text_facade.services.documents import QueryDocumentsService

@singleton
class QueryDocumentsHandler(RequestHandler[QueryDocumentsRequest, QueryDocumentsResponse]):

    @inject
    def __init__(self, query_documents_service: QueryDocumentsService) -> None:
        super().__init__()
        self._query_documents_service = query_documents_service

    def _on_validate(self, request: QueryDocumentsRequest) -> None:
        pass

    def _on_invoke(self, request: QueryDocumentsRequest) -> QueryDocumentsResponse:
        # Execute query
        results = self._query_documents_service.execute_query(
            index_name=request.index_name or "",
            query_text=request.query_text or "",
            max_results=request.max_results or 10,
            score_threshold=request.score_threshold or 1.0
        )

        # Return response
        return QueryDocumentsResponse(
            results=[
                QueryDocumentsResult(
                    entry_id=r.entry_id,
                    payload=r.payload,
                    score=r.score
                ) for r in results
            ]
        )