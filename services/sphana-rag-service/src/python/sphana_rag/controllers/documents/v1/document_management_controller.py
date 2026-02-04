from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_rag.controllers.documents.v1.handlers import DeleteDocumentHandler, IngestDocumentHandler, UpdateDocumentHandler, ListDocumentsHandler, GetDocumentHandler, DocumentExistsHandler

router = APIRouter(prefix="/v1/documents")

@router.post(":ingest")
async def ingest_index(request: Request, ingest_document_handler: IngestDocumentHandler = Injected(IngestDocumentHandler)) -> Response:
    return await ingest_document_handler.invoke(request)

@router.post(":update")
async def update_index(request: Request, update_document_handler: UpdateDocumentHandler = Injected(UpdateDocumentHandler)) -> Response:
    return await update_document_handler.invoke(request)

@router.post(":delete")
async def delete_index(request: Request, delete_document_handler: DeleteDocumentHandler = Injected(DeleteDocumentHandler)) -> Response:
    return await delete_document_handler.invoke(request)

@router.post(":get")
async def get_index(request: Request, get_document_handler: GetDocumentHandler = Injected(GetDocumentHandler)) -> Response:
    return await get_document_handler.invoke(request)

@router.post(":list")
async def list_indices(request: Request, list_documents_handler: ListDocumentsHandler = Injected(ListDocumentsHandler)) -> Response:
    return await list_documents_handler.invoke(request)

@router.post(":exists")
async def index_exists(request: Request, document_exists_handler: DocumentExistsHandler = Injected(DocumentExistsHandler)) -> Response:
    return await document_exists_handler.invoke(request)