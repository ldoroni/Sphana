import logging
import os
import torch
import sys
from pathlib import Path
from injector import singleton
from prometheus_client import Counter, Histogram
from sentence_transformers import SentenceTransformer
from time import time
from transformers import AutoTokenizer
from sphana_rag.models import TextChunkDetails

EMBEDDING_EXE_COUNTER = Counter("spn_embedding_exe_total", "Total number of embedding operations executed", ["operation"])
EMBEDDING_EXE_DURATION_HISTOGRAM = Histogram("spn_embedding_exe_duration_seconds", "Duration of embedding operations in seconds", ["operation"])

@singleton
class TextTokenizer:
    
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
        
        # Verify it's a valid model directory
        model_path = Path(local_model_path)
        required_files = ['config.json']
        missing_files = [f for f in required_files if not (model_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Local model directory exists but is incomplete; Missing files: {missing_files}; Path: {local_model_path}")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the tokenizer
        self.__tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True
        )
        
        # Load the embedding model
        self.__model = SentenceTransformer(
            local_model_path,
            device=device,
            trust_remote_code=True
        )
        
        self.__logger.info(f"TextTokenizer initialized with device: {device} and model {local_model_path}")

    def tokenize_text(self, text: str) -> list[float]:
        start_time: float = time()
        EMBEDDING_EXE_COUNTER.labels(operation="tokenize_text").inc()
        try:
            if not text or not text.strip():
                return []
            
            embedding = self.__model.encode(
                text,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            return embedding.tolist()
        finally:
            duration: float = time() - start_time
            EMBEDDING_EXE_DURATION_HISTOGRAM.labels(operation="tokenize_text").observe(duration)
    
    def tokenize_and_chunk_text(self, text: str, max_parent_chunk_size: int, max_child_chunk_size: int, parent_chunk_overlap_size: int, child_chunk_overlap_size: int) -> list[TextChunkDetails]:
        start_time: float = time()
        EMBEDDING_EXE_COUNTER.labels(operation="tokenize_and_chunk_text").inc()
        try:
            if not text or not text.strip():
                return []
            
            if max_parent_chunk_size <= 0:
                raise ValueError("max_parent_chunk_size must be greater than 0")
            
            if max_child_chunk_size <= 0:
                raise ValueError("max_child_chunk_size must be greater than 0")
            
            if max_child_chunk_size > max_parent_chunk_size:
                raise ValueError("max_child_chunk_size must be less than or equal to max_parent_chunk_size")
            
            if parent_chunk_overlap_size < 0:
                raise ValueError("parent_chunk_overlap_size must be non-negative")
            
            if parent_chunk_overlap_size >= max_parent_chunk_size:
                raise ValueError("parent_chunk_overlap_size must be less than max_parent_chunk_size")
            
            if child_chunk_overlap_size < 0:
                raise ValueError("child_chunk_overlap_size must be non-negative")
            
            if child_chunk_overlap_size >= max_child_chunk_size:
                raise ValueError("child_chunk_overlap_size must be less than max_child_chunk_size")
            
            # Tokenize the entire text
            encoding = self.__tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False
            )
            
            tokens = encoding['input_ids']
            offsets = encoding['offset_mapping']
            
            if not tokens:
                return []
            
            # First pass: create parent chunks with overlap
            parent_chunks = []
            start_idx = 0
            
            while start_idx < len(tokens):
                end_idx = min(start_idx + max_parent_chunk_size, len(tokens))
                
                parent_offsets = offsets[start_idx:end_idx]
                parent_start_char = parent_offsets[0][0]
                parent_end_char = parent_offsets[-1][1]
                parent_text = text[parent_start_char:parent_end_char]
                
                parent_chunks.append({
                    'text': parent_text,
                    'token_start': start_idx,
                    'token_end': end_idx,
                    'start_char': parent_start_char,
                    'end_char': parent_end_char
                })
                
                if end_idx >= len(tokens):
                    break
                
                start_idx = end_idx - parent_chunk_overlap_size
            
            # Second pass: subdivide each parent into non-overlapping child chunks
            child_chunks = []
            child_texts = []
            
            for parent in parent_chunks:
                child_start_idx = parent['token_start']
                
                while child_start_idx < parent['token_end']:
                    child_end_idx = min(child_start_idx + max_child_chunk_size, parent['token_end'])
                    
                    child_offsets = offsets[child_start_idx:child_end_idx]
                    child_start_char = child_offsets[0][0]
                    child_end_char = child_offsets[-1][1]
                    child_text = text[child_start_char:child_end_char]
                    
                    child_chunks.append({
                        'text': child_text,
                        'parent_text': parent['text'],
                        'token_count': child_end_idx - child_start_idx,
                        'start_char': child_start_char,
                        'end_char': child_end_char
                    })
                    
                    child_texts.append(child_text)
                    
                    if child_end_idx >= parent['token_end']:
                        break
                    
                    child_start_idx = child_end_idx - child_chunk_overlap_size
            
            # Third pass: generate embeddings for all child chunks in batch
            # Using "search_document" prefix for document embeddings as per nomic best practices
            prefixed_texts = [f"search_document: {child_text}" for child_text in child_texts]
            embeddings = self.__model.encode(
                prefixed_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # Fourth pass: create TextChunkDetails objects with embeddings
            text_chunks = []
            for i, chunk_data in enumerate(child_chunks):
                text_chunk = TextChunkDetails(
                    text=chunk_data['text'],
                    parent_text=chunk_data['parent_text'],
                    token_count=chunk_data['token_count'],
                    start_char=chunk_data['start_char'],
                    end_char=chunk_data['end_char'],
                    embedding=embeddings[i].tolist()
                )
                text_chunks.append(text_chunk)
            
            return text_chunks
        finally:
            duration: float = time() - start_time
            EMBEDDING_EXE_DURATION_HISTOGRAM.labels(operation="tokenize_and_chunk_text").observe(duration)