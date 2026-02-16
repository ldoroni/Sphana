import shutil
from abc import ABC
from prometheus_client import Counter, Histogram
from time import time
from typing import Optional
from rocksdict import Rdict, AccessType
from sphana_data_store.models import ListResults
from sphana_data_store.utils import Base64Util

ROCKSDB_EXE_COUNTER = Counter("spn_rocksdb_exe_total", "Total number of RocksDB operations executed", ["table", "operation"])
ROCKSDB_EXE_DURATION_HISTOGRAM = Histogram("spn_rocksdb_exe_duration_seconds", "Duration of RocksDB operations in seconds", ["table", "operation"])

class BaseDocumentsRepository[TDocument](ABC):

    def __init__(self, db_location: str, secondary: bool):
        self.__db_location: str = db_location
        self.__table_map: dict[str, Rdict] = {}
        self.__secondary: bool = secondary
        self.__secondary_path: str = "" # TODO: take from env variables - should be local path in the pod (not in PCV!)

    def _init_table(self, table_name: str) -> None:
        start_time: float = time()
        ROCKSDB_EXE_COUNTER.labels(table=table_name, operation="init_table").inc()
        try:
            self._get_table(table_name)
        finally:
            duration: float = time() - start_time
            ROCKSDB_EXE_DURATION_HISTOGRAM.labels(table=table_name, operation="init_table").observe(duration)

    def _drop_table(self, table_name: str) -> None:
        start_time: float = time()
        ROCKSDB_EXE_COUNTER.labels(table=table_name, operation="drop_table").inc()
        try:
            self.__drop_table(table_name)
        finally:
            duration: float = time() - start_time
            ROCKSDB_EXE_DURATION_HISTOGRAM.labels(table=table_name, operation="drop_table").observe(duration)

    def _upsert_document(self, table_name: str, document_id: str, document: TDocument) -> None:
        start_time: float = time()
        ROCKSDB_EXE_COUNTER.labels(table=table_name, operation="upsert_document").inc()
        try:
            table: Rdict = self._get_table(table_name)
            table.put(document_id, document)
        finally:
            duration: float = time() - start_time
            ROCKSDB_EXE_DURATION_HISTOGRAM.labels(table=table_name, operation="upsert_document").observe(duration)

    def _delete_document(self, table_name: str, document_id: str) -> None:
        start_time: float = time()
        ROCKSDB_EXE_COUNTER.labels(table=table_name, operation="delete_document").inc()
        try:
            table: Rdict = self._get_table(table_name)
            table.delete(document_id)
        finally:
            duration: float = time() - start_time
            ROCKSDB_EXE_DURATION_HISTOGRAM.labels(table=table_name, operation="delete_document").observe(duration)

    def _read_document(self, table_name: str, document_id: str) -> Optional[TDocument]:
        start_time: float = time()
        ROCKSDB_EXE_COUNTER.labels(table=table_name, operation="read_document").inc()
        try:
            table: Rdict = self._get_table(table_name)
            return table.get(document_id)
        finally:
            duration: float = time() - start_time
            ROCKSDB_EXE_DURATION_HISTOGRAM.labels(table=table_name, operation="read_document").observe(duration)

    def _list_documents(self, table_name: str, offset: Optional[str], limit: int) -> ListResults[TDocument]:
        start_time: float = time()
        ROCKSDB_EXE_COUNTER.labels(table=table_name, operation="list_documents").inc()
        try:
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
        finally:
            duration: float = time() - start_time
            ROCKSDB_EXE_DURATION_HISTOGRAM.labels(table=table_name, operation="list_documents").observe(duration)
    
    def _document_exists(self, table_name: str, document_id: str) -> bool:
        start_time: float = time()
        ROCKSDB_EXE_COUNTER.labels(table=table_name, operation="document_exists").inc()
        try:
            table: Rdict = self._get_table(table_name)
            return document_id in table
        finally:
            duration: float = time() - start_time
            ROCKSDB_EXE_DURATION_HISTOGRAM.labels(table=table_name, operation="document_exists").observe(duration)
    
    def _get_table(self, table_name: str) -> Rdict:
        if table_name in self.__table_map:
            table: Rdict = self.__table_map[table_name]
            if self.__secondary:
                table.try_catch_up_with_primary()
            return table

        # TODO: need lock
        table_location: str = self.__get_table_location(table_name)
        if self.__secondary:
            table: Rdict = Rdict(
                path=table_location, 
                access_type=AccessType.secondary(self.__secondary_path)
            )
        else:
            table: Rdict = Rdict(
                path=table_location,
                access_type=AccessType.read_write()
            )
        self.__table_map[table_name] = table
        return table

    def __drop_table(self, table_name: str) -> None:
        # TODO: if one pod dropped the index, the other pod will not know about that!
        db_location: str = self.__get_table_location(table_name)
        shutil.rmtree(db_location, ignore_errors=True)
        self.__table_map.pop(table_name, None)

    def __get_table_location(self, table_name: str) -> str:
        return f"{self.__db_location}/{table_name}"
    