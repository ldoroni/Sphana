from client_handler import ClientHandler
from injector import singleton
from .schemas import AddEmbeddingsRequest, AddEmbeddingsResponse

@singleton
class DataStoreEmbeddingsClient(ClientHandler):

    def __init__(self) -> None:
        super().__init__(host="http://127.0.0.1:5001/v1/embeddings")

    def add_embeddings(self, request: AddEmbeddingsRequest) -> AddEmbeddingsResponse:
        result = self.invoke(api="add", request=request.model_dump())
        return AddEmbeddingsResponse.model_validate(result)
