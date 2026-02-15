from typing import Optional
from pydantic import BaseModel
from .entry_details import EntryDetails

class GetEntryResponse(BaseModel):
    entry_details: Optional[EntryDetails] = None