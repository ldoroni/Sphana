import uuid
import numpy
import os
import shutil
from abc import ABC
from pathlib import Path
from prometheus_client import Counter, Histogram
from time import time
from typing import Optional
from faiss import IndexFlatL2, IndexIDMap2, write_index, read_index
from sphana_rag.models import TextChunkResult

FAISS_EXE_COUNTER = Counter("spn_faiss_exe_total", "Total number of Faiss operations executed", ["index", "operation"])
FAISS_DURATION_HISTOGRAM = Histogram("spn_faiss_exe_duration_seconds", "Duration of Faiss operations in seconds", ["index", "operation"])

class BaseVectorsRepository(ABC):
    
    def __init__(self, db_location: str, secondary: bool, dimension: int):
        self.__db_location: str = db_location
        self.__secondary: bool = secondary
        self.__dimension: int = dimension
        self.__index_map: dict[str, IndexIDMap2] = {}

    def _init_index(self, index_name: str) -> None:
        start_time: float = time()
        FAISS_EXE_COUNTER.labels(index=index_name, operation="init_index").inc()
        try:
            if not(index_name in self.__index_map):
                # TODO: check that it does not exist in the files system!
                quantizer = IndexFlatL2(self.__dimension)
                index = IndexIDMap2(quantizer)
                self.__index_map[index_name] = index
                self.__save_index(index_name, index)
        finally:
            duration: float = time() - start_time
            FAISS_DURATION_HISTOGRAM.labels(index=index_name, operation="init_index").observe(duration)

    def _drop_index(self, index_name: str) -> None:
        start_time: float = time()
        FAISS_EXE_COUNTER.labels(index=index_name, operation="drop_index").inc()
        try:
            self.__drop_index(index_name)
        finally:
            duration: float = time() - start_time
            FAISS_DURATION_HISTOGRAM.labels(index=index_name, operation="drop_index").observe(duration)

    def _ingest_chunk(self, index_name: str, chunk_id: str, chunk_vector: list[float]):
        start_time: float = time()
        FAISS_EXE_COUNTER.labels(index=index_name, operation="ingest_chunk").inc()
        try:
            index: IndexIDMap2 = self.__get_index(index_name)
            x = numpy.array([chunk_vector]).astype(numpy.float32)
            xids = numpy.array([int(chunk_id)]).astype(numpy.int64)
            index.add_with_ids(x, xids) # type: ignore
            self.__save_index(index_name, index)
        finally:
            duration: float = time() - start_time
            FAISS_DURATION_HISTOGRAM.labels(index=index_name, operation="ingest_chunk").observe(duration)

    def _delete_chunk(self, index_name: str, chunk_id: str) -> None:
        start_time: float = time()
        FAISS_EXE_COUNTER.labels(index=index_name, operation="delete_chunk").inc()
        try:
            index: IndexIDMap2 = self.__get_index(index_name)
            xids = numpy.array([int(chunk_id)]).astype(numpy.int64)
            index.remove_ids(xids) # type: ignore
            self.__save_index(index_name, index)
        finally:
            duration: float = time() - start_time
            FAISS_DURATION_HISTOGRAM.labels(index=index_name, operation="delete_chunk").observe(duration)

    def _search(self, index_name: str, query_vector: list[float], max_results: int) -> list[TextChunkResult]:
        start_time: float = time()
        FAISS_EXE_COUNTER.labels(index=index_name, operation="search").inc()
        try:
            index: IndexIDMap2 = self.__get_index(index_name)
            if index.ntotal == 0:
                return []
            xq = numpy.array([query_vector]).astype(numpy.float32)
            D, I = index.search(xq, max_results)  # type: ignore
            results: list[TextChunkResult] = []
            for distance, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                result: TextChunkResult = TextChunkResult(
                    chunk_id=str(idx), 
                    score=float(distance)
                )
                results.append(result)
            return results
        finally:
            duration: float = time() - start_time
            FAISS_DURATION_HISTOGRAM.labels(index=index_name, operation="search").observe(duration)

    def __get_index(self, index_name: str) -> IndexIDMap2:
        index: Optional[IndexIDMap2] = self.__index_map.get(index_name)
        if index is None:
            index = self.__load_index(index_name)
        return index
    
    def __save_index(self, index_name: str, index: IndexIDMap2) -> None:
        index_location: str = self.__get_index_location(index_name)
        # 0. Ensure directory exists
        index_path = Path(index_location)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        # 1. Write to a temporary file on the SAME volume
        index_location_tmp: str = f"{index_location}.{uuid.uuid4()}.tmp"
        write_index(index, index_location_tmp)
        # 2. Force the OS to flush data to the physical storage
        with open(index_location_tmp, 'a') as f:
            os.fsync(f.fileno())
        # 3. Atomically replace the old index with the new one
        os.replace(index_location_tmp, index_location)

    def __load_index(self, index_name: str) -> IndexIDMap2:
        index_location: str = self.__get_index_location(index_name)
        return read_index(index_location)
    
    def __drop_index(self, index_name: str) -> None:
        index_location: str = self.__get_index_location(index_name)
        index_path = Path(index_location)
        shutil.rmtree(index_path.parent, ignore_errors=True)

    def __get_index_location(self, index_name: str) -> str:
        return f"{self.__db_location}/{index_name}/db.index"
