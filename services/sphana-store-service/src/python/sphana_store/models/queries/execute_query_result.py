from pydantic import BaseModel

class ExecuteQueryResult(BaseModel):
    entry_id: str
    chunk_id: str
    payload: bytes
    score: float