import shutil
from abc import ABC
from typing import Optional
from rocksdict import Rdict
from sphana_rag.models import ListResults
from sphana_rag.utils import Base64Util

class BaseDocumentsRepository[TDocument](ABC):

    def __init__(self, db_location: str, secondary: bool):
        self.__db_location: str = db_location
        self.__table_map: dict[str, Rdict] = {}
        self.__secondary: bool = secondary

    def _init_table(self, table_name: str) -> None:
        self._get_table(table_name)

    def _drop_table(self, table_name: str) -> None:
        self.__drop_table(table_name)

    def _upsert_document(self, table_name: str, document_id: str, document: TDocument) -> None:
        table: Rdict = self._get_table(table_name)
        table.put(document_id, document)

    def _delete_document(self, table_name: str, document_id: str) -> None:
        table: Rdict = self._get_table(table_name)
        table.delete(document_id)

    def _read_document(self, table_name: str, document_id: str) -> Optional[TDocument]:
        table: Rdict = self._get_table(table_name)
        return table.get(document_id)

    def _list_documents(self, table_name: str, offset: Optional[str], limit: int) -> ListResults[TDocument]:
        table: Rdict = self._get_table(table_name)
        plain_offset: Optional[str] = Base64Util.from_nullable_base64(offset)
        items = table.items(from_key=plain_offset)
        completed: bool = True
        next_offset: Optional[str] = None
        documents: list[TDocument] = []
        for key, value in items:
            if len(documents) < limit:
                documents.append(value)
            else:
                next_offset = str(key)
                completed = False
                break
        return ListResults[TDocument](
            documents=documents, 
            next_offset=Base64Util.to_nullable_base64(next_offset), 
            completed=completed
        )
    
    def _document_exists(self, table_name: str, document_id: str) -> bool:
        table: Rdict = self._get_table(table_name)
        return document_id in table
    
    def _get_table(self, table_name: str) -> Rdict:
        if table_name in self.__table_map:
            table: Rdict = self.__table_map[table_name]
            if self.__secondary:
                table.try_catch_up_with_primary()
            return table

        # TODO: need lock
        table_location: str = self.__get_table_location(table_name)
        table: Rdict = Rdict(table_location)
        self.__table_map[table_name] = table
        return table

    def __drop_table(self, table_name: str) -> None:
        # TODO: if one pod dropped the index, the other pod will not know about that!
        db_location: str = self.__get_table_location(table_name)
        shutil.rmtree(db_location, ignore_errors=True)
        self.__table_map.pop(table_name, None)

    def __get_table_location(self, table_name: str) -> str:
        return f"{self.__db_location}/{table_name}"
    