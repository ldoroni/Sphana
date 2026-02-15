from injector import inject, singleton
from sphana_store.controllers.entries.v1.schemas import UpdateEntryRequest, UpdateEntryResponse
from sphana_store.services.entries import UpdateEntryService
from request_handler import RequestHandler

@singleton
class UpdateEntryHandler(RequestHandler[UpdateEntryRequest, UpdateEntryResponse]):

    @inject
    def __init__(self, 
                 update_entry_service: UpdateEntryService):
        super().__init__()
        self.__update_entry_service = update_entry_service

    def _on_validate(self, request: UpdateEntryRequest):
        # Validate request
        pass

    def _on_invoke(self, request: UpdateEntryRequest) -> UpdateEntryResponse:
        # Update entry
        self.__update_entry_service.update_entry(
            index_name=request.index_name or "",
            entry_id=request.entry_id or "",
            title=request.title,
            metadata=request.metadata
        )

        # Return response
        return UpdateEntryResponse()
