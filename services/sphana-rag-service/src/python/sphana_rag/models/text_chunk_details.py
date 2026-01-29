from pydantic import BaseModel, Field

class TextChunkDetails(BaseModel):
    text: str = Field(..., description="The chunk text content")
    token_count: int = Field(..., description="Number of tokens in this chunk")
    start_char: int = Field(..., description="Starting character position in the original text")
    end_char: int = Field(..., description="Ending character position in the original text")
    embedding: list[float] = Field(..., description="The embedding vector for this chunk")