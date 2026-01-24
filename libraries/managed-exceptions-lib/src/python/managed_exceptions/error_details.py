from http import HTTPStatus
from pydantic import BaseModel

class ErrorDetails(BaseModel):
    status_code: HTTPStatus
    diagnostic_code: str
    diagnostic_details: dict[str, str]
    message: str