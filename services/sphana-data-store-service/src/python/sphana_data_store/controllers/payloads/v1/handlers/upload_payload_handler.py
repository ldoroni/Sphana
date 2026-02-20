from injector import inject, singleton
from sphana_data_store.controllers.payloads.v1.schemas import UploadPayloadRequest, UploadPayloadResponse
from sphana_data_store.services.payloads import UploadPayloadService
from request_handler import RequestHandler

@singleton
class UploadPayloadHandler(RequestHandler[UploadPayloadRequest, UploadPayloadResponse]):
    
    @inject
    def __init__(self, 
                 upload_payload_service: UploadPayloadService):
        super().__init__()
        self.__upload_payload_service = upload_payload_service

    def _on_validate(self, request: UploadPayloadRequest):
        # Validate request
        pass

    def _on_invoke(self, request: UploadPayloadRequest) -> UploadPayloadResponse:
        # Upload payload
        self.__upload_payload_service.upload_payload(
            index_name=request.index_name or "",
            entry_id=request.entry_id or "",
            payload=request.payload or b"",
        )

        # Return response
        return UploadPayloadResponse()
