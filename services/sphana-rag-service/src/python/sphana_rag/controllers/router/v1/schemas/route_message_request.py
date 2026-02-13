from typing import Any
from pydantic import BaseModel

class RouteMessageRequest(BaseModel):
    topic_name: str
    shard_name: str
    message: dict[str, Any]
