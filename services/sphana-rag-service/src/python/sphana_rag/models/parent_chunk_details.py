from pydantic import BaseModel

class ParentChunkDetails(BaseModel):
    parent_chunk_id: str
    document_id: str
    chunk_index: int
    content: bytes