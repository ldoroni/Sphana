import logging
from typing import Any, Callable, Optional
from injector import inject, singleton
from distributed_cache import DistributedCache, DistributedTaskRouter, TaskResult
from managed_exceptions import InternalErrorException


@singleton
class ClusterRouterService:
    """
    Topic-based message router for write operations across the cluster.
    
    Services register local handlers via `listen(topic_name, callback)`.
    Callers use `route(shard_name, topic_name, message)` to dispatch 
    write operations to the correct node that owns the shard.
    
    Delegates to `DistributedTaskRouter` from the distributed-cache-lib,
    which uses the partition table to route messages.  When a node dies
    the coordinator detects the missing heartbeat, the rebalancer
    reassigns partitions to surviving nodes, and subsequent `route()`
    calls are automatically routed to the new owner — no manual
    intervention required.
    
    - If the target node is this pod, the registered callback is invoked directly.
    - If the target node is a remote pod, the message is forwarded via RPC.
    """

    @inject
    def __init__(self, distributed_cache: DistributedCache):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__task_router: DistributedTaskRouter = distributed_cache.get_task_router()

    def listen(self, topic_name: str, callback: Callable[[str, dict], Optional[dict]]) -> None:
        """
        Register a local handler for a topic.
        
        When a message is routed to this node for the given topic,
        the callback will be invoked with a dict message.
        
        Args:
            topic_name: The topic identifier (e.g., "shard.ingest_document").
            callback: A callable that accepts a shard name and a dict message, and returns a dict result or None.
        """
        self.__task_router.listen(topic_name, callback)

    def route(self, shard_name: str, topic_name: str, message: dict) -> Optional[dict]:
        """
        Route a message to the node that owns the given shard.
        
        Uses the distributed cache's partition table for routing.
        If the owning node is this pod, the local listener is invoked directly.
        If the owning node is a remote pod, the message is forwarded via RPC.
        
        If the target node is unavailable, the partition table is automatically
        updated by the coordinator (via heartbeat failure detection and
        rebalancing), and the request will be routed to the new owner on
        retry.
        
        Args:
            shard_name: The shard name used to determine the target node (routing key).
            topic_name: The topic identifier to dispatch the message to.
            message: A dict containing the message payload.
            
        Returns:
            A dict result from the listener callback (local or remote).
            
        Raises:
            InternalErrorException: If the task execution fails.
        """
        result: TaskResult = self.__task_router.submit(
            routing_key=shard_name,
            topic=topic_name,
            message=message,
            timeout=120.0,
            max_retries=2,
        )

        if result.is_success:
            return result.response
        
        self.__logger.error(
            f"Route failed for topic={topic_name}, shard={shard_name}, "
            f"node={result.node_address}: {result.error}"
        )
        raise InternalErrorException(
            message=f"Failed to route message to shard '{shard_name}': {result.error}",
        )

    def route_to_all(self, topic_name: str, message: dict) -> dict[str, Any]:
        """
        Broadcast a message to every node in the cluster.
        
        Useful for operations that must run on every pod, such as
        `create_index` or `delete_index`.
        
        Args:
            topic_name: The topic identifier to dispatch the message to.
            message: A dict containing the message payload.
            
        Returns:
            A dict mapping node_address → response dict.
            
        Raises:
            InternalErrorException: If any node fails to execute.
        """
        results: dict[str, TaskResult] = self.__task_router.submit_to_all(
            topic=topic_name,
            message=message,
            timeout=120.0,
        )

        responses: dict[str, Any] = {}
        errors: list[str] = []

        for node_address, result in results.items():
            if result.is_success:
                responses[node_address] = result.response
            else:
                errors.append(f"{node_address}: {result.error}")

        if errors:
            self.__logger.error(
                f"Route to all failed for topic={topic_name}: {errors}"
            )
            raise InternalErrorException(
                message=f"Failed to broadcast to all nodes: {'; '.join(errors)}",
            )

        return responses