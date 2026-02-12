from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails
from sphana_rag.repositories import IndexDetailsRepository, DocumentDetailsRepository
from sphana_rag.utils import ShardUtil

@singleton
class DocumentExistsService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 document_details_repository: DocumentDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__document_details_repository = document_details_repository

    def document_exists(self, index_name: str, document_id: str) -> bool:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Get shard name
        shard_name: str = ShardUtil.compute_shard_name(
            index_name, 
            document_id, 
            index_details.number_of_shards
        )
        
        # Get document existence
        return self.__document_details_repository.exists(shard_name, document_id)
