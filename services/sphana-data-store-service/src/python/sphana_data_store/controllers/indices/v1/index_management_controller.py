from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_data_store.controllers.indices.v1.handlers import CreateIndexHandler, DeleteIndexHandler, GetIndexHandler, IndexExistsHandler, ListIndicesHandler, UpdateIndexHandler

router = APIRouter(prefix="/v1/indices")

@router.post(":create")
async def create_index(request: Request, create_index_handler: CreateIndexHandler = Injected(CreateIndexHandler)) -> Response:
    return await create_index_handler.invoke(request)

@router.post(":update")
async def update_index(request: Request, update_index_handler: UpdateIndexHandler = Injected(UpdateIndexHandler)) -> Response:
    return await update_index_handler.invoke(request)

@router.post(":delete")
async def delete_index(request: Request, delete_index_handler: DeleteIndexHandler = Injected(DeleteIndexHandler)) -> Response:
    return await delete_index_handler.invoke(request)

@router.post(":get")
async def get_index(request: Request, get_index_handler: GetIndexHandler = Injected(GetIndexHandler)) -> Response:
    return await get_index_handler.invoke(request)

@router.post(":list")
async def list_indices(request: Request, list_indices_handler: ListIndicesHandler = Injected(ListIndicesHandler)) -> Response:
    return await list_indices_handler.invoke(request)

@router.post(":exists")
async def index_exists(request: Request, index_exists_handler: IndexExistsHandler = Injected(IndexExistsHandler)) -> Response:
    return await index_exists_handler.invoke(request)