from http import HTTPStatus
from managed_exceptions.error_details import ErrorDetails
from managed_exceptions.managed_exception import ManagedException

class ItemNotFoundException(ManagedException):
    def __init__(self, message: str, diagnostic_details: dict[str, str] = {}):
        super().__init__(ErrorDetails(
            status_code=HTTPStatus.NOT_FOUND,
            diagnostic_code="00404",
            diagnostic_details=diagnostic_details,
            message=message
        ))