import torch
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sphana_rag.models import TextChunkDetails

class TextTokenizer:
    """
    Tokenizer class for tokenizing text using the nomic-ai/nomic-embed-text-v1.5 model.
    Supports CUDA acceleration and provides text chunking functionality with token-based overlap.
    Designed to work with FastAPI's dependency injection system.
    """
    
    def __init__(self):
        """Initialize the tokenizer and model. Loads model eagerly at startup."""
        self._model_name = "nomic-ai/nomic-embed-text-v1.5"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        
        # Load the embedding model
        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
            trust_remote_code=True
        )
        
        print(f"TextTokenizer initialized with device: {self._device}")
    
    def chunk_text(
        self, 
        text: str, 
        max_chunk_size: int, 
        max_chunk_overlap_size: int
    ) -> list[TextChunkDetails]:
        """
        Tokenize and chunk text based on token limits with overlap, generating embeddings for each chunk.
        
        Args:
            text: The input text to be chunked
            max_chunk_size: Maximum number of tokens per chunk
            chunk_overlap_size: Number of tokens to overlap between consecutive chunks
            
        Returns:
            List of TextChunk objects, each containing:
                - text: The chunk text
                - token_count: Number of tokens in the chunk
                - start_char: Starting character position in original text
                - end_char: Ending character position in original text
                - embedding: The embedding vector for this chunk
        """
        if not text or not text.strip():
            return []
        
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0")
        
        if max_chunk_overlap_size < 0:
            raise ValueError("chunk_overlap_size must be non-negative")
        
        if max_chunk_overlap_size >= max_chunk_size:
            raise ValueError("chunk_overlap_size must be less than max_chunk_size")
        
        # Tokenize the entire text
        encoding = self._tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False
        )
        
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']
        
        if not tokens:
            return []
        
        chunks = []
        chunk_texts = []
        start_idx = 0
        
        # First pass: create chunks and collect texts
        while start_idx < len(tokens):
            # Determine the end index for this chunk
            end_idx = min(start_idx + max_chunk_size, len(tokens))
            
            # Extract chunk tokens and their offsets
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_offsets = offsets[start_idx:end_idx]
            
            # Get character positions
            start_char = chunk_offsets[0][0]
            end_char = chunk_offsets[-1][1]
            
            # Extract the actual text for this chunk
            chunk_text = text[start_char:end_char]
            
            # Store chunk metadata (without embedding for now)
            chunks.append({
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'start_char': start_char,
                'end_char': end_char
            })
            
            chunk_texts.append(chunk_text)
            
            # Move to the next chunk with overlap
            if end_idx >= len(tokens):
                break
            
            # Move forward by (max_chunk_size - chunk_overlap_size)
            start_idx = end_idx - max_chunk_overlap_size
        
        # Second pass: generate embeddings for all chunks in batch
        # Using "search_document" prefix for document embeddings as per nomic best practices
        prefixed_texts = [f"search_document: {chunk_text}" for chunk_text in chunk_texts]
        embeddings = self._model.encode(
            prefixed_texts,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        # Third pass: create TextChunkDetails objects with embeddings
        text_chunks = []
        for i, chunk_data in enumerate(chunks):
            text_chunk = TextChunkDetails(
                text=chunk_data['text'],
                token_count=chunk_data['token_count'],
                start_char=chunk_data['start_char'],
                end_char=chunk_data['end_char'],
                embedding=embeddings[i].tolist()
            )
            text_chunks.append(text_chunk)
        
        return text_chunks
    
    def get_device(self) -> str:
        """Return the device being used (cuda or cpu)."""
        return self._device
    
    def get_model_name(self) -> str:
        """Return the name of the loaded model."""
        return self._model_name
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The input text
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        encoding = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False
        )
        
        return len(encoding['input_ids'])

@lru_cache()
def get_text_tokenizer() -> TextTokenizer:
    """
    Factory function that returns a cached TextTokenizer instance.
    This ensures the model is loaded only once and reused across all requests.
    Use this with FastAPI's Depends() for dependency injection.
    
    Example:
        def my_service(tokenizer: TextTokenizer = Depends(get_text_tokenizer)):
            chunks = tokenizer.chunk_text(text, 512, 50)
    """
    return TextTokenizer()