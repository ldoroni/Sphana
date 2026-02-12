from injector import inject, singleton
from sphana_rag.controllers.indices.v1.schemas import DeleteIndexRequest, DeleteIndexResponse
from sphana_rag.services.indices import DeleteIndexService
from request_handler import RequestHandler

@singleton
class DeleteIndexHandler(RequestHandler[DeleteIndexRequest, DeleteIndexResponse]):

    @inject
    def __init__(self, 
                 delete_index_service: DeleteIndexService):
        super().__init__()
        self.__delete_index_service = delete_index_service

    def _on_validate(self, request: DeleteIndexRequest):
        # Validate request
        pass

    def _on_invoke(self, request: DeleteIndexRequest) -> DeleteIndexResponse:
        # Delete index
        self.__delete_index_service.delete_index(
            index_name=request.index_name or "",
        )

        # Return response
        return DeleteIndexResponse()
