from http import HTTPStatus
from managed_exceptions.error_details import ErrorDetails
from managed_exceptions.managed_exception import ManagedException

class ItemAlreadyExistsException(ManagedException):
    def __init__(self, message: str, diagnostic_details: dict[str, str] = {}):
        super().__init__(ErrorDetails(
            status_code=HTTPStatus.CONFLICT,
            diagnostic_code="00409",
            diagnostic_details=diagnostic_details,
            message=message
        ))