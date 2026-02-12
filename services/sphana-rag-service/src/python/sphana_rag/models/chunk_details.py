from pydantic import BaseModel

class ChildChunkDetails(BaseModel):
    child_chunk_id: str
    parent_chunk_id: str
    # document_id: str
    # child_chunk_index: int