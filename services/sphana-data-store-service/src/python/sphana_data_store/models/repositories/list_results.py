from typing import Optional
from pydantic import BaseModel

class ListResults[TDocument](BaseModel):
    documents: list[TDocument]
    next_offset: Optional[str] = None
    completed: bool = True