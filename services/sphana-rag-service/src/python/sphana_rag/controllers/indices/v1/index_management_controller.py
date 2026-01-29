from fastapi import APIRouter, Depends, Request, Response
from managed_exceptions import UnimplementedException
from sphana_rag.controllers.indices.v1.handlers import CreateIndexHandler

router = APIRouter(prefix="/v1/indices")

@router.post(":create")
async def create_index(request: Request, create_index_handler: CreateIndexHandler = Depends(CreateIndexHandler)) -> Response:
    return await create_index_handler.invoke(request)

@router.post(":update")
async def update_index(request: Request) -> Response:
    raise UnimplementedException("Method not implemented")

@router.post(":delete")
async def delete_index(request: Request) -> Response:
    raise UnimplementedException("Method not implemented")

@router.post(":get")
async def get_index(request: Request) -> Response:
    raise UnimplementedException("Method not implemented")

@router.post(":list")
async def list_indices(request: Request) -> Response:
    raise UnimplementedException("Method not implemented")

@router.post(":exists")
async def index_exists(request: Request) -> Response:
    raise UnimplementedException("Method not implemented")
