from client_handler import ClientHandler
from injector import singleton
from .schemas import UploadPayloadRequest, UploadPayloadResponse

@singleton
class DataStorePayloadsClient(ClientHandler):

    def __init__(self) -> None:
        super().__init__(host="http://127.0.0.1:5001/v1/payloads")

    def upload_payload(self, request: UploadPayloadRequest) -> UploadPayloadResponse:
        result = self.invoke(api="upload", request=request.model_dump())
        return UploadPayloadResponse.model_validate(result)
