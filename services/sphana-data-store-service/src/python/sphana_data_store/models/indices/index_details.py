from datetime import datetime
from pydantic import BaseModel

class IndexDetails(BaseModel):
    index_name: str
    description: str
    number_of_shards: int
    creation_timestamp: datetime
    modification_timestamp: datetime