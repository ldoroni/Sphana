from typing import Optional
from pydantic import BaseModel

class EmbedTextRequest(BaseModel):
    text: Optional[str] = None
