from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_rag.controllers.router.v1.handlers import RouteMessageHandler

router = APIRouter(prefix="/internal/v1/router")

@router.post(":route")
async def route_message(request: Request, route_message_handler: RouteMessageHandler = Injected(RouteMessageHandler)) -> Response:
    return await route_message_handler.invoke(request)