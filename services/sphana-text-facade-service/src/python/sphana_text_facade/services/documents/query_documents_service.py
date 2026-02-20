from injector import inject, singleton
from sphana_text_facade.clients.store import DataStoreQueriesClient
from sphana_text_facade.clients.store.schemas import ExecuteQueryRequest, ExecuteQueryResponse, ExecuteQueryResult
from sphana_text_facade.services.tokenizer import TextEmbedderService

@singleton
class QueryDocumentsService:

    @inject
    def __init__(self,
                 text_embedder_service: TextEmbedderService,
                 data_store_queries_client: DataStoreQueriesClient) -> None:
        self.__text_embedder_service = text_embedder_service
        self.__data_store_queries_client = data_store_queries_client

    def execute_query(self, index_name: str, query_text: str, max_results: int, score_threshold: float) -> list[ExecuteQueryResult]:
        # Step 1: Embed the query text
        query_embedding: list[float] = self.__text_embedder_service.embed_text(text=query_text)

        # Step 2: Execute vector similarity query against data store
        execute_query_request = ExecuteQueryRequest(
            index_name=index_name,
            query_embedding=query_embedding,
            max_results=max_results,
            score_threshold=score_threshold
        )
        execute_query_response: ExecuteQueryResponse = self.__data_store_queries_client.execute_query(execute_query_request)
        return execute_query_response.results