from pydantic import BaseModel

class EmbeddingResult(BaseModel):
    embedding_id: str
    score: float