from injector import inject, singleton
from sphana_data_store.controllers.entries.v1.schemas import CreateEntryRequest, CreateEntryResponse
from sphana_data_store.services.entries import CreateEntryService
from request_handler import RequestHandler

@singleton
class CreateEntryHandler(RequestHandler[CreateEntryRequest, CreateEntryResponse]):
    
    @inject
    def __init__(self, 
                 create_entry_service: CreateEntryService):
        super().__init__()
        self.__create_entry_service = create_entry_service

    def _on_validate(self, request: CreateEntryRequest):
        # Validate request
        pass

    def _on_invoke(self, request: CreateEntryRequest) -> CreateEntryResponse:
        # Create entry
        self.__create_entry_service.create_entry(
            index_name=request.index_name or "",
            entry_id=request.entry_id or "",
            title=request.title or "",
            metadata=request.metadata or {}
        )

        # Return response
        return CreateEntryResponse()
