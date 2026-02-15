from injector import inject, singleton
from sphana_store.controllers.entries.v1.schemas import ListEntriesRequest, ListEntriesResponse, EntryDetails
from sphana_store.services.entries import ListEntriesService
from request_handler import RequestHandler

@singleton
class ListEntriesHandler(RequestHandler[ListEntriesRequest, ListEntriesResponse]):

    @inject
    def __init__(self, 
                 list_entries_service: ListEntriesService):
        super().__init__()
        self.__list_entries_service = list_entries_service

    def _on_validate(self, request: ListEntriesRequest):
        # Validate request
        pass

    def _on_invoke(self, request: ListEntriesRequest) -> ListEntriesResponse:
        # List entries
        results = self.__list_entries_service.list_entries(
            index_name=request.index_name or "",
            offset=request.offset,
            limit=request.limit or 0
        )

        # Return response
        return ListEntriesResponse(
            entries_details=[
                EntryDetails(
                    entry_id=entry_details.entry_id,
                    title=entry_details.title,
                    metadata=entry_details.metadata,
                    creation_timestamp=entry_details.creation_timestamp,
                    modification_timestamp=entry_details.modification_timestamp
                ) for entry_details in results.documents
            ],
            next_offset=results.next_offset,
            completed=results.completed
        )
