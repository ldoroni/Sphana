import hashlib
from injector import inject, singleton

@singleton
class ShardUtil:
    
    @inject
    def __init__(self):
        pass

    def compute_shard_name(self, index_name: str, document_id: str, number_of_shards: int) -> str:
        partition_key: str = f"{index_name}/{document_id}"
        hash_value: int = int(hashlib.sha256(partition_key.encode("utf-8")).hexdigest(), 16)
        shard_number: int = hash_value % number_of_shards
        return self.get_shard_name(index_name, shard_number)
    
    def get_shard_name(self, index_name: str, shard_number: int) -> str:
        return f"{index_name}_{shard_number:06d}"
