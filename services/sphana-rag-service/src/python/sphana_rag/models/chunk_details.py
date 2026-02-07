from pydantic import BaseModel

class ChunkDetails(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    content: bytes