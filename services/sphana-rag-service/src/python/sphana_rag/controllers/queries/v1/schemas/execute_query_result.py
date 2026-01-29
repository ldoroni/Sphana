from pydantic import BaseModel

class ExecuteQueryResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
