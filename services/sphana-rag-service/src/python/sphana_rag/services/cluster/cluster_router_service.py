import httpx
import logging
from http import HTTPStatus
from typing import Any, Callable, Optional
from injector import inject, singleton
from managed_exceptions import UpstreamException
from sphana_rag.configs import RagConfig
from .cluster_nodes_service import ClusterNodesService

@singleton
class ClusterRouterService:
    """
    Topic-based message router for write operations across the cluster.
    
    Services register local handlers via `listen(topic_name, callback)`.
    Callers use `route(shard_name, topic_name, message)` to dispatch 
    write operations to the correct node that owns the shard.
    
    All messages and responses are dicts (JSON-serializable).
    
    - If the target node is this pod, the registered callback is invoked directly.
    - If the target node is a remote pod, the message is forwarded via HTTP POST 
      to the remote node's internal route endpoint.
    """
    
    INTERNAL_ROUTE_PATH: str = "/internal/v1/router:route"

    @inject
    def __init__(self, 
                 rag_config: RagConfig,
                 cluster_nodes_service: ClusterNodesService):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__rag_config = rag_config
        self.__cluster_nodes_service = cluster_nodes_service
        self.__listeners: dict[str, Callable[[str, dict], Optional[dict]]] = {}
        self.__http_client = httpx.Client(timeout=120.0)

    def listen(self, topic_name: str, callback: Callable[[str, dict], Optional[dict]]) -> None:
        """
        Register a local handler for a topic.
        
        When a message is routed to this node for the given topic,
        the callback will be invoked with a dict message.
        
        Args:
            topic_name: The topic identifier (e.g., "shard.ingest_document").
            callback: A callable that accepts a shard name and a dict message, and returns a dict result or None.
        """
        if topic_name in self.__listeners:
            self.__logger.warning(f"Overwriting existing listener for topic: {topic_name}")
        self.__listeners[topic_name] = callback
        self.__logger.info(f"Registered listener for topic: {topic_name}")

    def route(self, shard_name: str, topic_name: str, message: dict) -> Optional[dict]:
        """
        Route a message to the node that owns the given shard.
        
        If the owning node is this pod, the local listener is invoked directly.
        If the owning node is a remote pod, the message is forwarded via HTTP.
        
        Args:
            shard_name: The shard name used to determine the target node.
            topic_name: The topic identifier to dispatch the message to.
            message: A dict containing the message payload.
            
        Returns:
            A dict result from the listener callback (local or remote).
            
        Raises:
            ValueError: If no listener is registered for the topic (local dispatch).
        """
        # Determine which node owns this shard
        target_node: str = self.__cluster_nodes_service.get_node(shard_name)
        self_node: str = self.__rag_config.self_node

        if target_node == self_node:
            # Local dispatch
            self.__logger.debug(f"Local dispatch for topic={topic_name}, shard={shard_name}")
            return self._invoke_local(topic_name, shard_name, message)
        else:
            # Remote dispatch
            self.__logger.debug(f"Remote dispatch to {target_node} for topic={topic_name}, shard={shard_name}")
            return self._invoke_remote(target_node, topic_name, shard_name, message)

    def _invoke_local(self, topic_name: str, shard_name: str, message: dict) -> Optional[dict]:
        """
        Invoke the locally registered callback for the given topic.
        
        Args:
            topic_name: The topic identifier.
            shard_name: The shard name associated with the message.
            message: The dict message payload.
            
        Returns:
            A dict result from the callback.
            
        Raises:
            ValueError: If no listener is registered for the topic.
        """
        callback = self.__listeners.get(topic_name)
        if callback is None:
            raise ValueError(f"No listener registered for topic: {topic_name}")
        return callback(shard_name, message)

    def _invoke_remote(self, node_url: str, topic_name: str, shard_name: str, message: dict) -> Optional[dict]:
        """
        Forward a message to a remote node via HTTP POST.
        
        The remote node's internal route endpoint will look up its own
        local listener for the topic and invoke it.
        
        Args:
            node_url: The base URL of the target node (e.g., "http://node-2:5001").
            topic_name: The topic identifier.
            shard_name: The shard name associated with the message.
            message: The dict message payload.
            
        Returns:
            A dict result from the remote node.
            
        Raises:
            httpx.HTTPStatusError: If the remote node returns a non-2xx status.
        """
        url: str = f"{node_url.rstrip('/')}{self.INTERNAL_ROUTE_PATH}"
        self.__logger.info(f"Forwarding message to {url} for topic={topic_name}")
        
        # Prepare payload
        payload: dict[str, Any] = {
            "topic_name": topic_name,
            "shard_name": shard_name,
            "message": message
        }
        
        # Execute HTTP request
        response = self.__http_client.post(url, json=payload)

        # Parse response
        response_data: dict = response.json()
        if response.status_code == 200:
            return response_data.get("response", {})
        else:
            raise UpstreamException(
                http_status=HTTPStatus(response.status_code),
                message=response_data.get("message", ""),
                diagnostic_code=response_data.get("diagnostic_code", ""),
                diagnostic_details=response_data.get("diagnostic_details", {}),
            )