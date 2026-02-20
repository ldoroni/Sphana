from injector import inject, singleton
from sphana_data_store.controllers.payloads.v1.schemas import AppendPayloadRequest, AppendPayloadResponse
from sphana_data_store.services.payloads import AppendPayloadService
from sphana_data_store.utils import Base64Util
from request_handler import RequestHandler

@singleton
class AppendPayloadHandler(RequestHandler[AppendPayloadRequest, AppendPayloadResponse]):
    
    @inject
    def __init__(self, 
                 append_payload_service: AppendPayloadService):
        super().__init__()
        self.__append_payload_service = append_payload_service

    def _on_validate(self, request: AppendPayloadRequest):
        # Validate request
        pass

    def _on_invoke(self, request: AppendPayloadRequest) -> AppendPayloadResponse:
        # Upload payload
        self.__append_payload_service.append_payload(
            index_name=request.index_name or "",
            entry_id=request.entry_id or "",
            payload=request.payload or b""
        )

        # Return response
        return AppendPayloadResponse()
