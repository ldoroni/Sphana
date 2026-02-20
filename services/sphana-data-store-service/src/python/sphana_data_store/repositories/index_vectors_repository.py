from injector import singleton
from sphana_data_store.models import EmbeddingResult
from .base_vectors_repository import BaseVectorsRepository

@singleton
class IndexVectorsRepository(BaseVectorsRepository):
    
    def __init__(self):
        db_location: str = "./.database/index_vectors" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        dimension = 768 # TODO: take from index details!
        super().__init__(db_location, secondary, dimension)

    def init_index(self, shard_name: str) -> None:
        self._init_index(shard_name)

    def drop_index(self, shard_name: str) -> None:
        self._drop_index(shard_name)

    def add(self, shard_name: str, embedding_id: str, embedding_vector: list[float]):
        self._add_embedding(shard_name, embedding_id, embedding_vector)

    def delete(self, shard_name: str, embedding_id: str) -> None:
        self._delete_embedding(shard_name, embedding_id)

    def search(self, shard_name: str, query_vector: list[float], max_results: int) -> list[EmbeddingResult]:
        return self._search(shard_name, query_vector, max_results)
