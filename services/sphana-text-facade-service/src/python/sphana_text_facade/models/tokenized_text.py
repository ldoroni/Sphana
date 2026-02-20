from pydantic import BaseModel, Field

class TokenizedText(BaseModel):
    text: str = Field(..., description="The text content of this chunk")
    token_ids: list[int] = Field(..., description="Token IDs from the AutoTokenizer")
    offsets: list[tuple[int, int]] = Field(..., description="Character offset mappings for each token (start, end)")
    start_char: int = Field(..., description="Character index in the original text where this chunk starts")
    end_char: int = Field(..., description="Character index in the original text where this chunk ends")