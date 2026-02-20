from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_data_store.controllers.payloads.v1.handlers import AppendPayloadHandler, UploadPayloadHandler, DeletePayloadHandler

router = APIRouter(prefix="/v1/payloads")

@router.post(":append")
async def append_payload(request: Request, append_payload_handler: AppendPayloadHandler = Injected(AppendPayloadHandler)) -> Response:
    return await append_payload_handler.invoke(request)

@router.post(":upload")
async def upload_payload(request: Request, upload_payload_handler: UploadPayloadHandler = Injected(UploadPayloadHandler)) -> Response:
    return await upload_payload_handler.invoke(request)

@router.post(":delete")
async def delete_payload(request: Request, delete_payload_handler: DeletePayloadHandler = Injected(DeletePayloadHandler)) -> Response:
    return await delete_payload_handler.invoke(request)
