from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChildChunkDetailsRepository, ParentChunkDetailsRepository
from sphana_rag.utils import ShardUtil

@singleton
class DeleteIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 child_chunk_details_repository: ChildChunkDetailsRepository,
                 parent_chunk_details_repository: ParentChunkDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository

    def delete_index(self, index_name: str) -> None:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")

        for shard_number in range(index_details.number_of_shards):
            # Get shard name
            shard_name: str = ShardUtil.get_shard_name(index_name, shard_number)

            # Drop parent chunk details table
            self.__parent_chunk_details_repository.drop_table(shard_name)

            # Drop child chunk details table
            self.__child_chunk_details_repository.drop_table(shard_name)

            # Drop document details table
            self.__document_details_repository.drop_table(shard_name)

            # Drop index vectors index
            self.__index_vectors_repository.drop_index(shard_name)

        # Delete index details
        self.__index_details_repository.delete(index_name)