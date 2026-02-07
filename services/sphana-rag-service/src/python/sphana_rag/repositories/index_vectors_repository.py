from injector import singleton
import numpy
import shutil
from pathlib import Path
from typing import Optional
from faiss import IndexFlatL2, IndexIDMap2, write_index, read_index
from sphana_rag.models import TextChunkResult

@singleton
class IndexVectorsRepository:
    
    def __init__(self):
        self.__db_location: str = "./.database/index_vectors_db" # TODO: take from env variables
        self.__db_map: dict[str, IndexIDMap2] = {}
        self.__dimension = 768 # TODO: take from env variables
        self.__secondary: bool = False # TODO: take from env variables

    def init_index(self, shard_name: str) -> None:
        if not(shard_name in self.__db_map):
            # TODO: check that it does not exist in the files system!
            quantizer = IndexFlatL2(self.__dimension)
            index = IndexIDMap2(quantizer)
            self.__db_map[shard_name] = index
            self.__save_index(shard_name, index)

    def drop_index(self, shard_name: str) -> None:
        self.__drop_index(shard_name)

    def ingest(self, shard_name: str, chunk_id: str, chunk_vector: list[float]):
        index: IndexIDMap2 = self.__get_index(shard_name)
        x = numpy.array([chunk_vector]).astype(numpy.float32)
        xids = numpy.array([int(chunk_id)]).astype(numpy.int64)
        index.add_with_ids(x, xids) # type: ignore
        self.__save_index(shard_name, index)

    def delete(self, shard_name: str, chunk_id: str) -> None:
        index: IndexIDMap2 = self.__get_index(shard_name)
        xids = numpy.array([int(chunk_id)]).astype(numpy.int64)
        index.remove_ids(xids) # type: ignore
        self.__save_index(shard_name, index)

    def search(self, shard_name: str, query_vector: list[float], max_results: int) -> list[TextChunkResult]:
        index: IndexIDMap2 = self.__get_index(shard_name)
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

    def __get_index(self, shard_name: str) -> IndexIDMap2:
        index: Optional[IndexIDMap2] = self.__db_map.get(shard_name)
        if index is None:
            index = self.__load_index(shard_name)
        return index
    
    def __save_index(self, shard_name: str, index: IndexIDMap2) -> None:
        index_location: str = self.__get_index_location(shard_name)
        index_path = Path(index_location)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        write_index(index, index_location)

    def __load_index(self, shard_name: str) -> IndexIDMap2:
        index_location: str = self.__get_index_location(shard_name)
        return read_index(index_location)
    
    def __drop_index(self, shard_name: str) -> None:
        index_location: str = self.__get_index_location(shard_name)
        index_path = Path(index_location)
        shutil.rmtree(index_path.parent, ignore_errors=True)

    def __get_index_location(self, shard_name: str) -> str:
        return f"{self.__db_location}/{shard_name}/db.index"
