# """
# Test script for TextTokenizer class.
# Run this after installing dependencies with: pip install -e .
# """

# from sphana_rag.services.tokenizer import TextTokenizer, get_text_tokenizer, TextChunk

# def test_basic_chunking():
#     """Test basic text chunking functionality."""
#     print("=" * 80)
#     print("Testing TextTokenizer")
#     print("=" * 80)
    
#     # Initialize the tokenizer using the factory function
#     tokenizer = get_text_tokenizer()
#     print(f"\nModel: {tokenizer.get_model_name()}")
#     print(f"Device: {tokenizer.get_device()}")
    
#     # Sample text
#     sample_text = """
#     Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural 
#     intelligence displayed by humans and animals. Leading AI textbooks define the field as the study 
#     of "intelligent agents": any device that perceives its environment and takes actions that maximize 
#     its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" 
#     is often used to describe machines (or computers) that mimic "cognitive" functions that humans 
#     associate with the human mind, such as "learning" and "problem solving".
#     """
    
#     print(f"\n{'='*80}")
#     print("Sample Text:")
#     print(f"{'='*80}")
#     print(sample_text.strip())
    
#     # Count tokens
#     token_count = tokenizer.count_tokens(sample_text)
#     print(f"\n{'='*80}")
#     print(f"Total tokens: {token_count}")
#     print(f"{'='*80}")
    
#     # Chunk the text
#     max_chunk_size = 50
#     chunk_overlap_size = 10
    
#     print(f"\nChunking with max_chunk_size={max_chunk_size}, chunk_overlap_size={chunk_overlap_size}")
#     chunks = tokenizer.chunk_text(sample_text, max_chunk_size, chunk_overlap_size)
    
#     print(f"\n{'='*80}")
#     print(f"Generated {len(chunks)} chunks:")
#     print(f"{'='*80}")
    
#     for i, chunk in enumerate(chunks, 1):
#         print(f"\nChunk {i}:")
#         print(f"  Token count: {chunk.token_count}")
#         print(f"  Character range: {chunk.start_char}-{chunk.end_char}")
#         print(f"  Embedding length: {len(chunk.embedding)}")
#         print(f"  Embedding sample (first 5): {chunk.embedding[:5]}")
#         print(f"  Text: {chunk.text[:100]}..." if len(chunk.text) > 100 else f"  Text: {chunk.text}")
    
#     print(f"\n{'='*80}")
#     print("Test completed successfully!")
#     print(f"{'='*80}")

# def test_cached_factory():
#     """Test that the factory function returns the same cached instance."""
#     print("\n" + "="*80)
#     print("Testing Cached Factory Pattern (FastAPI DI)")
#     print("="*80)
    
#     tokenizer1 = get_text_tokenizer()
#     tokenizer2 = get_text_tokenizer()
    
#     assert tokenizer1 is tokenizer2, "Factory function should return the same cached instance!"
#     print("✓ Cached factory confirmed: both calls return the same instance")
#     print("  This ensures the model is loaded only once when used with FastAPI's Depends()")

# def test_edge_cases():
#     """Test edge cases."""
#     print("\n" + "="*80)
#     print("Testing Edge Cases")
#     print("="*80)
    
#     tokenizer = get_text_tokenizer()
    
#     # Empty text
#     chunks = tokenizer.chunk_text("", 100, 10)
#     assert len(chunks) == 0, "Empty text should return empty list"
#     print("✓ Empty text handled correctly")
    
#     # Single token text
#     chunks = tokenizer.chunk_text("Hello", 100, 10)
#     assert len(chunks) == 1, "Single word should return one chunk"
#     assert isinstance(chunks[0], TextChunk), "Chunk should be a TextChunk instance"
#     assert len(chunks[0].embedding) > 0, "Chunk should have an embedding"
#     print("✓ Single token text handled correctly")
#     print(f"  Embedding dimension: {len(chunks[0].embedding)}")
    
#     # Small chunk size
#     chunks = tokenizer.chunk_text("This is a test sentence.", 5, 2)
#     assert len(chunks) > 1, "Small chunk size should create multiple chunks"
#     assert all(isinstance(chunk, TextChunk) for chunk in chunks), "All chunks should be TextChunk instances"
#     assert all(len(chunk.embedding) > 0 for chunk in chunks), "All chunks should have embeddings"
#     print(f"✓ Small chunk size created {len(chunks)} chunks")
    
#     print("\nAll edge cases passed!")

# def test_text_chunk_model():
#     """Test the TextChunk Pydantic model."""
#     print("\n" + "="*80)
#     print("Testing TextChunk Model")
#     print("="*80)
    
#     # Create a sample chunk
#     chunk = TextChunk(
#         text="Sample text",
#         token_count=2,
#         start_char=0,
#         end_char=11,
#         embedding=[0.1, 0.2, 0.3]
#     )
    
#     print(f"✓ TextChunk created successfully")
#     print(f"  Text: {chunk.text}")
#     print(f"  Token count: {chunk.token_count}")
#     print(f"  Char range: {chunk.start_char}-{chunk.end_char}")
#     print(f"  Embedding: {chunk.embedding}")
    
#     # Test JSON serialization
#     chunk_json = chunk.model_dump_json()
#     print(f"✓ TextChunk can be serialized to JSON")
#     print(f"  JSON length: {len(chunk_json)} characters")
    
#     # Test JSON deserialization
#     chunk_restored = TextChunk.model_validate_json(chunk_json)
#     assert chunk_restored.text == chunk.text, "Restored chunk should match original"
#     print(f"✓ TextChunk can be deserialized from JSON")

# def test_fastapi_usage_example():
#     """Demonstrate how to use with FastAPI dependency injection."""
#     print("\n" + "="*80)
#     print("FastAPI Usage Example")
#     print("="*80)
    
#     print("""
# To use TextTokenizer with FastAPI dependency injection:

# from fastapi import Depends
# from sphana_rag.services.tokenizer import TextTokenizer, TextChunk, get_text_tokenizer

# class MyService:
#     def __init__(self, tokenizer: TextTokenizer = Depends(get_text_tokenizer)):
#         self.tokenizer = tokenizer
    
#     def process_document(self, text: str) -> list[TextChunk]:
#         chunks = self.tokenizer.chunk_text(
#             text, 
#             max_chunk_size=512, 
#             chunk_overlap_size=50
#         )
        
#         # Each chunk now contains:
#         # - chunk.text: The text content
#         # - chunk.token_count: Number of tokens
#         # - chunk.start_char/end_char: Character positions
#         # - chunk.embedding: The embedding vector (list[float])
        
#         return chunks

# # The tokenizer will be loaded once at startup and reused across all requests
# # thanks to the @lru_cache() decorator on get_text_tokenizer()
#     """)

# if __name__ == "__main__":
#     try:
#         test_basic_chunking()
#         test_cached_factory()
#         test_edge_cases()
#         test_text_chunk_model()
#         test_fastapi_usage_example()
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()