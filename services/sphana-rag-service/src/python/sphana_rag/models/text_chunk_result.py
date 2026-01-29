from pydantic import BaseModel

class TextChunkResult(BaseModel):
    chunk_id: str
    score: float