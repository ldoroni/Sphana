from fastapi import APIRouter, Depends, Request, Response
from managed_exceptions import UnimplementedException
from sphana_rag.controllers.documents.v1.handlers import IngestDocumentHandler

router = APIRouter(prefix="/v1/documents")

@router.post(":ingest")
async def ingest_index(request: Request, ingest_document_handler: IngestDocumentHandler = Depends(IngestDocumentHandler)) -> Response:
    return await ingest_document_handler.invoke(request)

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
