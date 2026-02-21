from pydantic import BaseModel, Base64Str

class ExecuteQueryResult(BaseModel):
    entry_id: str
    payload: Base64Str
    score: float
