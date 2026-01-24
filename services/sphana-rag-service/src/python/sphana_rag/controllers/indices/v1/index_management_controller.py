from fastapi import APIRouter, Depends, Request, Response
from fastapi_utils.cbv import cbv
from managed_exceptions import UnimplementedException
from sphana_rag.controllers.indices.v1.handlers import CreateIndexHandler

router = APIRouter(prefix="/v1/indices")

@cbv(router)
class IndexManagementController:
    def __init__(self, 
                 create_index_handler: CreateIndexHandler = Depends(CreateIndexHandler)):
        self.__create_index_handler = create_index_handler

    @router.post(":create")
    async def create_index(self, request: Request) -> Response:
        return await self.__create_index_handler.invoke(request)

    @router.post(":update")
    async def update_index(self, request: Request) -> Response:
        raise UnimplementedException("Method not implemented")

    @router.post(":delete")
    async def delete_index(self, request: Request) -> Response:
        raise UnimplementedException("Method not implemented")

    @router.post(":get")
    async def get_index(self, request: Request) -> Response:
        raise UnimplementedException("Method not implemented")

    @router.post(":list")
    async def list_indices(self, request: Request) -> Response:
        raise UnimplementedException("Method not implemented")

    @router.post(":exists")
    async def index_exists(self, request: Request) -> Response:
        raise UnimplementedException("Method not implemented")
