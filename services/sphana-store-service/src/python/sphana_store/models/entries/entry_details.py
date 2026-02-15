from datetime import datetime
from pydantic import BaseModel

class EntryDetails(BaseModel):
    entry_id: str
    title: str
    metadata: dict[str, str]
    creation_timestamp: datetime
    modification_timestamp: datetime