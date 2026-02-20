from client_handler import ClientHandler
from injector import singleton
from .schemas import CreateEntryRequest, CreateEntryResponse

@singleton
class DataStoreEntriesClient(ClientHandler):

    def __init__(self) -> None:
        super().__init__(host="http://localhost:5001/v1/entries")

    def create_entry(self, request: CreateEntryRequest) -> CreateEntryResponse:
        result = self.invoke(api="create", request=request.model_dump())
        return CreateEntryResponse.model_validate(result)
