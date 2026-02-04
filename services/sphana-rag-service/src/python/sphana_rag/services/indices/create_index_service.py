from datetime import datetime, timezone
from injector import inject, singleton
from managed_exceptions import ItemAlreadyExistsException
from sphana_rag.models import IndexDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ChunkDetailsRepository

@singleton
class CreateIndexService:
    
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

    def create_index(self, index_name: str, description: str, max_chunk_size: int, chunk_overlap_size: int) -> None:
        # Assert index name
        if self.__index_details_repository.exists(index_name):
            raise ItemAlreadyExistsException(f"Index {index_name} already exists")

        # Init chunk details table
        self.__chunk_details_repository.init_table(index_name)

        # Init document details table
        self.__document_details_repository.init_table(index_name)

        # Init index vectors index
        self.__index_vectors_repository.init_index(index_name)

        # Save index details
        index_details: IndexDetails = IndexDetails(
            index_name=index_name,
            description=description,
            max_chunk_size=max_chunk_size,
            chunk_overlap_size=chunk_overlap_size,
            creation_timestamp=datetime.now(timezone.utc),
            modification_timestamp=datetime.now(timezone.utc)
        )
        self.__index_details_repository.upsert(index_details)