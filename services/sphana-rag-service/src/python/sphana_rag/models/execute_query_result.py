from pydantic import BaseModel

class ExecuteQueryResult(BaseModel):
    document_id: str
    chunk_index: int
    content: bytes
    score: float