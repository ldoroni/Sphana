from injector import inject, singleton
from sphana_store.controllers.entries.v1.schemas import DeleteEntryRequest, DeleteEntryResponse
from sphana_store.services.entries import DeleteEntryService
from request_handler import RequestHandler

@singleton
class DeleteEntryHandler(RequestHandler[DeleteEntryRequest, DeleteEntryResponse]):

    @inject
    def __init__(self, 
                 delete_entry_service: DeleteEntryService):
        super().__init__()
        self.__delete_entry_service = delete_entry_service

    def _on_validate(self, request: DeleteEntryRequest):
        # Validate request
        pass

    def _on_invoke(self, request: DeleteEntryRequest) -> DeleteEntryResponse:
        # Delete entry
        self.__delete_entry_service.delete_entry(
            index_name=request.index_name or "",
            entry_id=request.entry_id or ""
        )

        # Return response
        return DeleteEntryResponse()
