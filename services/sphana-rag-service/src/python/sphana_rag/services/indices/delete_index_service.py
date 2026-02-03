from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChunkDetailsRepository

@singleton
class DeleteIndexService:
    
    @inject
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 chunk_details_repository: ChunkDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__chunk_details_repository = chunk_details_repository

    def delete_index(self, index_name: str) -> None:
        # Assert index name
        if not self.__index_details_repository.exists(index_name):
            raise ItemNotFoundException(f"Index {index_name} does not exist")

        # Drop chunk details table
        self.__chunk_details_repository.drop_table(index_name)

        # Drop document details table
        self.__document_details_repository.drop_table(index_name)

        # Drop index vectors index
        self.__index_vectors_repository.drop_index(index_name)

        # Delete index details
        self.__index_details_repository.delete(index_name)