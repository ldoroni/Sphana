from pydantic import BaseModel

class ExecuteQueryResult(BaseModel):
    entry_id: str
    payload: bytes
    score: float
