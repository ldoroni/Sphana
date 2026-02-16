from pydantic import BaseModel, Field

class TokenizedText(BaseModel):
    text: str = Field(..., description="The text content of this chunk")
    token_ids: list[int] = Field(..., description="Token IDs from the AutoTokenizer")
    offsets: list[tuple[int, int]] = Field(..., description="Character offset mappings for each token (start, end)")