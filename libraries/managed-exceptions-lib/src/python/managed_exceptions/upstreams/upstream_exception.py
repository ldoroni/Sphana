from http import HTTPStatus
from managed_exceptions.error_details import ErrorDetails
from managed_exceptions.managed_exception import ManagedException

class UpstreamException(ManagedException):
    def __init__(self, http_status: HTTPStatus, message: str, diagnostic_code: str, diagnostic_details: dict[str, str] = {}):
        super().__init__(ErrorDetails(
            status_code=http_status,
            diagnostic_code=diagnostic_code,
            diagnostic_details=diagnostic_details,
            message=message
        ))