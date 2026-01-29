from pathlib import Path
from typing import Optional
from faiss import IndexFlatL2, IndexIDMap2, write_index, read_index

class IndexVectorsRepository:
    def __init__(self):
        self.__db_location: str = "./.database/index_vectors_db" # TODO: take from env variables
        self.__indices_map: dict[str, IndexIDMap2] = {}
        self.__dimension = 1024 # TODO: take from env variables

    def init(self, index_name: str) -> None:
        if not(index_name in self.__indices_map):
            quantizer = IndexFlatL2(self.__dimension)
            index = IndexIDMap2(quantizer)
            self.__indices_map[index_name] = index
            self.__save_index(index_name, index)

    def __get_or_load_index(self, index_name: str) -> IndexIDMap2:
        index: Optional[IndexIDMap2] = self.__indices_map.get(index_name)
        if index == None:
            index = self.__load_index(index_name)
        return index
    
    def __save_index(self, index_name: str, index: IndexIDMap2) -> None:
        index_location: str = self.__get_index_location(index_name)
        index_path = Path(index_location)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        write_index(index, index_location)

    def __load_index(self, index_name: str) -> IndexIDMap2:
        index_location: str = self.__get_index_location(index_name)
        return read_index(index_location)

    def __get_index_location(self, index_name: str) -> str:
        return f"{self.__db_location}/{index_name}.index"
