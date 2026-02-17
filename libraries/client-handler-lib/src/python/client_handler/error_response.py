from pydantic import BaseModel

class ErrorResponse(BaseModel):
    diagnostic_code: str
    diagnostic_details: dict[str, str]
    message: str