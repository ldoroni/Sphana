from datetime import datetime, timezone
from fastapi import Depends
from managed_exceptions import ItemAlreadyExistsException
from sphana_rag.models import IndexDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository

class CreateIndexService:
    def __init__(self, 
                 index_details_repository: IndexDetailsRepository = Depends(IndexDetailsRepository),
                 index_vectors_repository: IndexVectorsRepository = Depends(IndexVectorsRepository)):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository

    def create_index(self, index_name: str, description: str, max_chunk_size: int, max_chunk_overlap_size: int) -> None:
        if self.__index_details_repository.exists(index_name):
            raise ItemAlreadyExistsException(f"Index {index_name} already exists")

        # Init index vectors DB
        self.__index_vectors_repository.init(index_name)

        # Save index details
        index_details: IndexDetails = IndexDetails(
            index_name=index_name,
            description=description,
            max_chunk_size=max_chunk_size,
            max_chunk_overlap_size=max_chunk_overlap_size,
            creation_timestamp=datetime.now(timezone.utc),
            modification_timestamp=datetime.now(timezone.utc)
        )
        self.__index_details_repository.upsert(index_details)