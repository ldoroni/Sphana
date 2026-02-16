from fastapi import APIRouter, Request, Response
from fastapi_injector import Injected
from sphana_data_store.controllers.entries.v1.handlers import DeleteEntryHandler, CreateEntryHandler, UpdateEntryHandler, ListEntriesHandler, GetEntryHandler, EntryExistsHandler

router = APIRouter(prefix="/v1/entries")

@router.post(":create")
async def create_entry(request: Request, create_entry_handler: CreateEntryHandler = Injected(CreateEntryHandler)) -> Response:
    return await create_entry_handler.invoke(request)

@router.post(":update")
async def update_entry(request: Request, update_entry_handler: UpdateEntryHandler = Injected(UpdateEntryHandler)) -> Response:
    return await update_entry_handler.invoke(request)

@router.post(":delete")
async def delete_entry(request: Request, delete_entry_handler: DeleteEntryHandler = Injected(DeleteEntryHandler)) -> Response:
    return await delete_entry_handler.invoke(request)

@router.post(":get")
async def get_entry(request: Request, get_entry_handler: GetEntryHandler = Injected(GetEntryHandler)) -> Response:
    return await get_entry_handler.invoke(request)

@router.post(":list")
async def list_entries(request: Request, list_entries_handler: ListEntriesHandler = Injected(ListEntriesHandler)) -> Response:
    return await list_entries_handler.invoke(request)

@router.post(":exists")
async def entry_exists(request: Request, entry_exists_handler: EntryExistsHandler = Injected(EntryExistsHandler)) -> Response:
    return await entry_exists_handler.invoke(request)