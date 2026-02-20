from injector import inject, singleton
from sphana_data_store.controllers.payloads.v1.schemas import DeletePayloadRequest, DeletePayloadResponse
from sphana_data_store.services.payloads import DeletePayloadService
from request_handler import RequestHandler

@singleton
class DeletePayloadHandler(RequestHandler[DeletePayloadRequest, DeletePayloadResponse]):
    
    @inject
    def __init__(self, 
                 delete_payload_service: DeletePayloadService):
        super().__init__()
        self.__delete_payload_service = delete_payload_service

    def _on_validate(self, request: DeletePayloadRequest):
        # Validate request
        pass

    def _on_invoke(self, request: DeletePayloadRequest) -> DeletePayloadResponse:
        # Delete payload
        self.__delete_payload_service.delete_payload(
            index_name=request.index_name or "",
            entry_id=request.entry_id or ""
        )

        # Return response
        return DeletePayloadResponse()
