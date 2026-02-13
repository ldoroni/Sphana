from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, DocumentDetails
from sphana_rag.repositories import IndexDetailsRepository, IndexVectorsRepository, DocumentDetailsRepository, ParentChunkDetailsRepository, ChildChunkDetailsRepository
from sphana_rag.services.cluster import ClusterRouterService
from sphana_rag.utils import ShardUtil

TOPIC_DELETE_DOCUMENT = "shard.delete_document"

@singleton
class DeleteDocumentService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 index_vectors_repository: IndexVectorsRepository,
                 document_details_repository: DocumentDetailsRepository,
                 parent_chunk_details_repository: ParentChunkDetailsRepository,
                 child_chunk_details_repository: ChildChunkDetailsRepository,
                 cluster_router_service: ClusterRouterService):
        self.__index_details_repository = index_details_repository
        self.__index_vectors_repository = index_vectors_repository
        self.__document_details_repository = document_details_repository
        self.__parent_chunk_details_repository = parent_chunk_details_repository
        self.__child_chunk_details_repository = child_chunk_details_repository
        self.__cluster_router_service = cluster_router_service

        # Register listener for shard write operations
        self.__cluster_router_service.listen(TOPIC_DELETE_DOCUMENT, self._handle_delete_writes)

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
        
        # Route write operations to the shard owner
        message: dict = {
            "index_name": index_name,
            "document_id": document_id
        }
        self.__cluster_router_service.route(shard_name, TOPIC_DELETE_DOCUMENT, message)

    def _handle_delete_writes(self, shard_name: str, message: dict) -> Optional[dict]:
        # Get message payload
        index_name: str = message["index_name"]
        document_id: str = message["document_id"]
        
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
        return None
