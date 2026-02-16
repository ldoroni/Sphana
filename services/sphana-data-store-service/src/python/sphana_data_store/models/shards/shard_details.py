from datetime import datetime
from pydantic import BaseModel

class ShardDetails(BaseModel):
    shard_name: str
    index_name: str
    creation_timestamp: datetime
    modification_timestamp: datetime