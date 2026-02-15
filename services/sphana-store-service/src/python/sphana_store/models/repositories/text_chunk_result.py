from pydantic import BaseModel

class TextChunkResult(BaseModel):
    embedding_id: str
    score: float