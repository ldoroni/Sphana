from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_data_store.controllers.embeddings.v1.handlers import AddEmbeddingsHandler, ResetEmbeddingsHandler

router = APIRouter(prefix="/v1/embeddings")

@router.post(":add")
async def add_embeddings(request: Request, add_embeddings_handler: AddEmbeddingsHandler = Injected(AddEmbeddingsHandler)) -> Response:
    return await add_embeddings_handler.invoke(request)

@router.post(":reset")
async def reset_embeddings(request: Request, reset_embeddings_handler: ResetEmbeddingsHandler = Injected(ResetEmbeddingsHandler)) -> Response:
    return await reset_embeddings_handler.invoke(request)
