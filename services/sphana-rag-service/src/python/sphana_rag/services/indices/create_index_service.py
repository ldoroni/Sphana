class CreateIndexService:
    def __init__(self):
        pass

    def create_index(self, index_name: str, description: str, max_chunk_size: int, max_chunk_overlap_size: int) -> None:
        print("create index service!!")