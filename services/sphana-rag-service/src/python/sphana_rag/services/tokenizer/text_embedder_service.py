import logging
import os
import sys
from pathlib import Path
from injector import singleton
from prometheus_client import Counter, Histogram
from time import time
from sentence_transformers import SentenceTransformer

EMBEDDER_EXE_COUNTER = Counter("spn_embedder_exe_total", "Total number of embedder operations executed", ["operation"])
EMBEDDER_EXE_DURATION_HISTOGRAM = Histogram("spn_embedder_exe_duration_seconds", "Duration of embedder operations in seconds", ["operation"])

@singleton
class TextEmbedderService:
    
    def __init__(self):
        self.__logger = logging.getLogger(self.__class__.__name__)
        
        # Calculate absolute path to local model
        # From: services/sphana-rag-service/src/python/sphana_rag/__main__.py
        # To:   services/sphana-rag-service/src/resources/models/embedding
        current_file = Path(sys.argv[0]).resolve()
        service_root = current_file.parents[2]
        local_model_path = str(service_root / "resources" / "models" / "embedding")
        
        # Verify local model exists
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model not found at: {local_model_path}")
        
        # Load the embedding model
        self.__embedding_model = SentenceTransformer(
            local_model_path,
            trust_remote_code=True
        )
        
        self.__logger.info(f"TokenEmbedder initialized with model {local_model_path}")

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a vector.
        Use for query embedding (with "search_query: " prefix).
        """
        embedding_list: list[list[float]] = self.embed_texts([text])
        return embedding_list[0] if embedding_list else []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings into vectors in batch.
        Use for document chunk embedding (with "search_document: " prefix).
        """
        start_time: float = time()
        EMBEDDER_EXE_COUNTER.labels(operation="embed_texts").inc()
        try:
            if not texts:
                return []
            embeddings = self.__embedding_model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return [embedding.tolist() for embedding in embeddings]
        finally:
            duration: float = time() - start_time
            EMBEDDER_EXE_DURATION_HISTOGRAM.labels(operation="embed_texts").observe(duration)