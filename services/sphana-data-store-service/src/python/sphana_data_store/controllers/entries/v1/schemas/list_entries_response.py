from typing import Optional
from pydantic import BaseModel
from .entry_details import EntryDetails

class ListEntriesResponse(BaseModel):
    entries_details: Optional[list[EntryDetails]] = None
    next_offset: Optional[str] = None
    completed: Optional[bool] = None
