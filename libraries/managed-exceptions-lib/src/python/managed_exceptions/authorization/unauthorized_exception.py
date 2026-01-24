from http import HTTPStatus
from managed_exceptions.error_details import ErrorDetails
from managed_exceptions.managed_exception import ManagedException

class UnauthorizedException(ManagedException):
    def __init__(self, message: str, diagnostic_details: dict[str, str] = {}):
        super().__init__(ErrorDetails(
            status_code=HTTPStatus.FORBIDDEN,
            diagnostic_code="00403",
            diagnostic_details=diagnostic_details,
            message=message
        ))