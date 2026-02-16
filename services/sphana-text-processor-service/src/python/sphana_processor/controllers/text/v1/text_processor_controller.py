from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_text_processor.controllers.text.v1.handlers import EmbedTextHandler, ChunkAndEmbedTextHandler

router = APIRouter(prefix="/v1/text")

@router.post(":embed")
async def embed_text(request: Request, embed_text_handler: EmbedTextHandler = Injected(EmbedTextHandler)) -> Response:
    return await embed_text_handler.invoke(request)

@router.post(":chunk_and_embed")
async def chunk_and_embed_text(request: Request, chunk_and_embed_text_handler: ChunkAndEmbedTextHandler = Injected(ChunkAndEmbedTextHandler)) -> Response:
    return await chunk_and_embed_text_handler.invoke(request)