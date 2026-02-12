from injector import singleton
from sphana_rag.models import TextChunkResult
from .base_vectors_repository import BaseVectorsRepository

@singleton
class IndexVectorsRepository(BaseVectorsRepository):
    
    def __init__(self):
        db_location: str = "./.database/index_vectors" # TODO: take from env variables
        secondary: bool = False # TODO: take from env variables
        dimension = 768 # TODO: take from env variables
        super().__init__(db_location, secondary, dimension)

    def init_index(self, shard_name: str) -> None:
        self._init_index(shard_name)

    def drop_index(self, shard_name: str) -> None:
        self._drop_index(shard_name)

    def ingest(self, shard_name: str, chunk_id: str, chunk_vector: list[float]):
        self._ingest_chunk(shard_name, chunk_id, chunk_vector)

    def delete(self, shard_name: str, chunk_id: str) -> None:
        self._delete_chunk(shard_name, chunk_id)

    def search(self, shard_name: str, query_vector: list[float], max_results: int) -> list[TextChunkResult]:
        return self._search(shard_name, query_vector, max_results)
