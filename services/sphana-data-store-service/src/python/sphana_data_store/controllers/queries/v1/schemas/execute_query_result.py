from pydantic import BaseModel, Base64Bytes

class ExecuteQueryResult(BaseModel):
    entry_id: str
    payload: Base64Bytes
    score: float
