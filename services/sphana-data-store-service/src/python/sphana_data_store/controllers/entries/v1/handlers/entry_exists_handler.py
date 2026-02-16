from injector import inject, singleton
from sphana_data_store.controllers.entries.v1.schemas import EntryExistsRequest, EntryExistsResponse
from sphana_data_store.services.entries import EntryExistsService
from request_handler import RequestHandler

@singleton
class EntryExistsHandler(RequestHandler[EntryExistsRequest, EntryExistsResponse]):

    @inject
    def __init__(self, 
                 entry_exists_service: EntryExistsService):
        super().__init__()
        self.__entry_exists_service = entry_exists_service

    def _on_validate(self, request: EntryExistsRequest):
        # Validate request
        pass

    def _on_invoke(self, request: EntryExistsRequest) -> EntryExistsResponse:
        # Check if entry exists
        exists: bool = self.__entry_exists_service.entry_exists(
            index_name=request.index_name or "",
            entry_id=request.entry_id or ""
        )

        # Return response
        return EntryExistsResponse(
            exists=exists
        )
