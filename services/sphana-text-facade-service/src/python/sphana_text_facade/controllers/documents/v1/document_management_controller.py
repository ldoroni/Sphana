from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_text_facade.controllers.documents.v1.handlers import IngestDocumentHandler, QueryDocumentsHandler

router = APIRouter(prefix="/v1/documents")

@router.post(":ingest")
async def ingest_document(request: Request, ingest_document_handler: IngestDocumentHandler = Injected(IngestDocumentHandler)) -> Response:
    return await ingest_document_handler.invoke(request)

@router.post(":query")
async def query_documents(request: Request, query_documents_handler: QueryDocumentsHandler = Injected(QueryDocumentsHandler)) -> Response:
    return await query_documents_handler.invoke(request)