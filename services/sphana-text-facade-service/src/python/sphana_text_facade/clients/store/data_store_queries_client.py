from client_handler import ClientHandler
from injector import singleton
from .schemas import ExecuteQueryRequest, ExecuteQueryResponse

@singleton
class DataStoreQueriesClient(ClientHandler):

    def __init__(self) -> None:
        super().__init__(host="http://127.0.0.1:5001/v1/queries")

    def execute_query(self, request: ExecuteQueryRequest) -> ExecuteQueryResponse:
        result = self.invoke(api="execute", request=request.model_dump())
        return ExecuteQueryResponse.model_validate(result)
