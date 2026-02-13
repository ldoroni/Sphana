from typing import Optional
from injector import inject, singleton
from sphana_rag.controllers.router.v1.schemas import RouteMessageRequest, RouteMessageResponse
from sphana_rag.services.cluster import ClusterRouterService
from request_handler import RequestHandler

@singleton
class RouteMessageHandler(RequestHandler[RouteMessageRequest, RouteMessageResponse]):

    @inject
    def __init__(self, 
                 cluster_router_service: ClusterRouterService):
        super().__init__()
        self.__cluster_router_service = cluster_router_service

    def _on_validate(self, request: RouteMessageRequest):
        # Validate request
        pass

    def _on_invoke(self, request: RouteMessageRequest) -> RouteMessageResponse:
        # Handle message
        response: Optional[dict] = self.__cluster_router_service._invoke_local(
            topic_name=request.topic_name,
            shard_name=request.shard_name,
            message=request.message
        )

        # Return response
        return RouteMessageResponse(
            response=response
        )
