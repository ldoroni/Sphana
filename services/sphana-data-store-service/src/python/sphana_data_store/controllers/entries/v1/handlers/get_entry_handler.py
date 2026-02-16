from injector import inject, singleton
from sphana_data_store.controllers.entries.v1.schemas import GetEntryRequest, GetEntryResponse, EntryDetails
from sphana_data_store.services.entries import GetEntryService
from request_handler import RequestHandler

@singleton
class GetEntryHandler(RequestHandler[GetEntryRequest, GetEntryResponse]):

    @inject
    def __init__(self, 
                 get_entry_service: GetEntryService):
        super().__init__()
        self.__get_entry_service = get_entry_service

    def _on_validate(self, request: GetEntryRequest):
        # Validate request
        pass

    def _on_invoke(self, request: GetEntryRequest) -> GetEntryResponse:
        # Get entry
        entry_details = self.__get_entry_service.get_entry(
            index_name=request.index_name or "",
            entry_id=request.entry_id or ""
        )

        # Return response
        return GetEntryResponse(
            entry_details=EntryDetails(
                entry_id=entry_details.entry_id,
                title=entry_details.title,
                metadata=entry_details.metadata,
                creation_timestamp=entry_details.creation_timestamp,
                modification_timestamp=entry_details.modification_timestamp
            )
        )
