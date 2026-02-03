from injector import inject, singleton
from sphana_rag.repositories import IndexDetailsRepository

@singleton
class IndexExistsService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository):
        self.__index_details_repository = index_details_repository

    def index_exists(self, index_name: str) -> bool:
        return self.__index_details_repository.exists(index_name)