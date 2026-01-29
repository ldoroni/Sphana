from pydantic import BaseModel
from .execute_query_result import ExecuteQueryResult

class ExecuteQueryResponse(BaseModel):
    results: list[ExecuteQueryResult]
