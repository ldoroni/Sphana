from typing import Optional
from pydantic import BaseModel

class RouteMessageResponse(BaseModel):
    response: Optional[dict]
