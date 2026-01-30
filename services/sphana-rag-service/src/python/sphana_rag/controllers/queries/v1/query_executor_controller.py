from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_rag.controllers.queries.v1.handlers import ExecuteQueryHandler

router = APIRouter(prefix="/v1/queries")

@router.post(":execute")
async def execute_query(request: Request, execute_query_handler: ExecuteQueryHandler = Injected(ExecuteQueryHandler)) -> Response:
    return await execute_query_handler.invoke(request)
