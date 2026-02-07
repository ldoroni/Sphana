import base64
from typing import Optional
from injector import inject, singleton
from managed_exceptions import ItemNotFoundException
from sphana_rag.models import IndexDetails, DocumentDetails, ListResults
from sphana_rag.repositories import IndexDetailsRepository, DocumentDetailsRepository
from sphana_rag.utils import ShardUtil, Base64Util

@singleton
class ListDocumentsService:
    
    @inject
    def __init__(self,
                 index_details_repository: IndexDetailsRepository,
                 document_details_repository: DocumentDetailsRepository):
        self.__index_details_repository = index_details_repository
        self.__document_details_repository = document_details_repository

    def list_documents(self, index_name: str,  offset: Optional[str], limit: int) -> ListResults[DocumentDetails]:
        # Get index details
        index_details: Optional[IndexDetails] = self.__index_details_repository.read(index_name)
        if index_details is None:
            raise ItemNotFoundException(f"Index {index_name} does not exist")
        
        # Compute starting shard id and actual offset
        # Note that there is a possibility that the after the "." separator there is no offset,
        # this happens when we have completed to iterate all documents in a shard, but still have more shards to iterate, 
        # so we move to the next shard with no offset
        start_shard_number = 0
        actual_offset = None
        if offset is not None:
            try:
                tokens: list[str] = offset.split('.')
                if len(tokens) > 0:
                    start_shard_number_str: str = Base64Util.from_base64(tokens[0])
                    start_shard_number = int(start_shard_number_str)
                    if len(tokens) > 1:
                        actual_offset = tokens[1]
            except Exception as e:
                pass
        
        # Iterate over shards and aggregate results
        documents: list[DocumentDetails] = []
        next_offset: Optional[str] = None
        completed: bool = True
        last_shard_number: int = -1
        for shard_number in range(start_shard_number, index_details.number_of_shards):
            last_shard_number = shard_number

            # Get shard name
            shard_name: str = ShardUtil.get_shard_name(index_name, shard_number)
            
            # List documents details
            actual_limit = limit - len(documents)
            shard_results: ListResults[DocumentDetails] = self.__document_details_repository.list(
                shard_name, 
                actual_offset, 
                actual_limit
            )

            # Aggregate results
            documents.extend(shard_results.documents)
            next_offset = shard_results.next_offset
            completed = shard_results.completed

            if len(documents) >= limit:
                break

        # Return results
        actual_next_offset: Optional[str]
        actual_completed = completed and last_shard_number == index_details.number_of_shards - 1
        if actual_completed:
            # Complete to iterate all shards, no more documents to return
            actual_next_offset = None
        elif not completed:
            # Still have documents in the same last shard, keep the same shard id in the offset
            last_shard_number_str: str = Base64Util.to_base64(str(last_shard_number))
            actual_next_offset = f"{last_shard_number_str}.{next_offset or ''}"
        else:
            # Completed to iterate the last shard, but still have more shards to iterate, move to the next shard in the offset
            next_shard_number = last_shard_number + 1
            next_shard_number_str: str = Base64Util.to_base64(str(next_shard_number))
            actual_next_offset = f"{next_shard_number_str}.{next_offset or ''}"

        return ListResults(
            documents=documents, 
            next_offset=actual_next_offset, 
            completed=actual_completed
        )
