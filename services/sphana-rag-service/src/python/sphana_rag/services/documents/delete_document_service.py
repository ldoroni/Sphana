from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, DocumentDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ParentChunkDetailsRepository, ChildChunkDetailsRepository, DistributedCacheRepository
from sphana_rag.utils import ShardUtil

@singleton
class DeleteDocumentService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 parent_chunk_details_repository: ParentChunkDetailsRepository,
                 child_chunk_details_repository: ChildChunkDetailsRepository,
                 distributed_cache_repository: DistributedCacheRepository):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__distributed_cache_repository = distributed_cache_repository

    def delete_document(self, index_name: str, document_id: str):
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
        
        # Delete information from the repository
        with self.__distributed_cache_repository.lock(shard_name, ttl_seconds=300):
            # Get document details
            document_details: Optional[DocumentDetails] = self.__document_details_repository.read(shard_name, document_id)
            if document_details is None:
                raise ItemNotFoundException(f"Document {document_id} does not exist in index {index_name}")
            
            # Delete parent chunks
            for parent_chunk_id in document_details.parent_chunk_ids:
                self.__parent_chunk_details_repository.delete(shard_name, parent_chunk_id)

            # Delete child chunks and their vectors
            for child_chunk_id in document_details.child_chunk_ids:
                self.__child_chunk_details_repository.delete(shard_name, child_chunk_id)
                self.__index_vectors_repository.delete(shard_name, child_chunk_id)

            # Delete document details
            self.__document_details_repository.delete(shard_name, document_id)