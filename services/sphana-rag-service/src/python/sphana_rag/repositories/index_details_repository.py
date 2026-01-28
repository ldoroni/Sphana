from typing import Optional
from rocksdict import Rdict
from sphana_rag.models import IndexDetails

class IndexDetailsRepository:
    def __init__(self):
        self.__db_location: str = "./.database/index_details_db" # TODO: take from env variables
        self.__db: Rdict = Rdict(self.__db_location)
        self.__secondary: bool = False # TODO: take from env variables

    def upsert(self, index_details: IndexDetails) -> None:
        if self.__secondary:
            self.__db.try_catch_up_with_primary()
        self.__db.put(index_details.index_name, index_details)

    def delete(self, index_name: str) -> None:
        if self.__secondary:
            self.__db.try_catch_up_with_primary()
        self.__db.delete(index_name)

    def read(self, index_name: str) -> Optional[IndexDetails]:
        if self.__secondary:
            self.__db.try_catch_up_with_primary()
        return self.__db.get(index_name)
    
    def exists(self, index_name: str) -> bool:
        if self.__secondary:
            self.__db.try_catch_up_with_primary()
        return index_name in self.__db

