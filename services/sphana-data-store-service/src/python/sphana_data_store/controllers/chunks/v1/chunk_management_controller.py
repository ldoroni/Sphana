from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_data_store.controllers.chunks.v1.handlers import IngestChunkHandler, ClearChunksHandler

router = APIRouter(prefix="/v1/entries")

@router.post(":ingest")
async def ingest_chunk(request: Request, ingest_chunk_handler: IngestChunkHandler = Injected(IngestChunkHandler)) -> Response:
    return await ingest_chunk_handler.invoke(request)

@router.post(":clear")
async def clear_chunks(request: Request, clear_chunks_handler: ClearChunksHandler = Injected(ClearChunksHandler)) -> Response:
    return await clear_chunks_handler.invoke(request)
