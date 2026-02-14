from distributed_cache import DistributedCache, ClusterNodesProvider, LockHandle
from injector import singleton

class K8sProvider(ClusterNodesProvider):
    def get_nodes(self) -> list[str]:
        # Return current pod IP:port list from Kubernetes API
        return ["http://127.0.0.1:5001"]
    
    def get_self_address(self) -> str:
        # Return this pod's IP:port address
        return "http://127.0.0.1:5002"

@singleton
class DistributedCacheRepository:

    def __init__(self):
        self.__cache = DistributedCache(
            self_address="http://127.0.0.1:5002",
            cluster_nodes_provider=K8sProvider(),
        )
        self.__cache.start()

    def lock(self, key: str, ttl_seconds: int) -> LockHandle:
        return self.__cache.acquire_lock(key, ttl_seconds)