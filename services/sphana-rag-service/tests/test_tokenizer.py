# """
# Test script for TextTokenizer class.
# Run this after installing dependencies with: pip install -e .
# """

# from sphana_rag.services.tokenizer import TextTokenizer
# from sphana_rag.models import TextChunkDetails

# def test_basic_parent_child_chunking():
#     """Test basic parent-child text chunking functionality."""
#     print("=" * 80)
#     print("Testing TextTokenizer - Parent-Child Chunking")
#     print("=" * 80)
    
#     # Initialize the tokenizer
#     tokenizer = TextTokenizer()
    
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
    
#     # Chunk the text with parent-child approach
#     max_parent_chunk_size = 100
#     max_child_chunk_size = 25
#     parent_chunk_overlap_size = 10
#     child_chunk_overlap_size = 5
    
#     print(f"\nChunking with max_parent_chunk_size={max_parent_chunk_size}, max_child_chunk_size={max_child_chunk_size}, parent_chunk_overlap_size={parent_chunk_overlap_size}, child_chunk_overlap_size={child_chunk_overlap_size}")
#     chunks = tokenizer.tokenize_and_chunk_text(
#         sample_text,
#         max_parent_chunk_size=max_parent_chunk_size,
#         max_child_chunk_size=max_child_chunk_size,
#         parent_chunk_overlap_size=parent_chunk_overlap_size,
#         child_chunk_overlap_size=child_chunk_overlap_size
#     )
    
#     print(f"\n{'='*80}")
#     print(f"Generated {len(chunks)} child chunks:")
#     print(f"{'='*80}")
    
#     for i, chunk in enumerate(chunks, 1):
#         print(f"\nChild Chunk {i}:")
#         print(f"  Token count: {chunk.token_count}")
#         print(f"  Character range: {chunk.start_char}-{chunk.end_char}")
#         print(f"  Embedding length: {len(chunk.embedding)}")
#         print(f"  Embedding sample (first 5): {chunk.embedding[:5]}")
#         print(f"  Child text: {chunk.text[:80]}..." if len(chunk.text) > 80 else f"  Child text: {chunk.text}")
#         print(f"  Parent text length: {len(chunk.parent_text)} chars")
#         print(f"  Parent text: {chunk.parent_text[:80]}..." if len(chunk.parent_text) > 80 else f"  Parent text: {chunk.parent_text}")
    
#     # Verify parent text is always larger or equal to child text
#     for chunk in chunks:
#         assert len(chunk.parent_text) >= len(chunk.text), "Parent text should be >= child text"
    
#     print(f"\n{'='*80}")
#     print("Test completed successfully!")
#     print(f"{'='*80}")

# def test_edge_cases():
#     """Test edge cases."""
#     print("\n" + "="*80)
#     print("Testing Edge Cases")
#     print("="*80)
    
#     tokenizer = TextTokenizer()
    
#     # Empty text
#     chunks = tokenizer.tokenize_and_chunk_text("", max_parent_chunk_size=100, max_child_chunk_size=25, parent_chunk_overlap_size=10, child_chunk_overlap_size=5)
#     assert len(chunks) == 0, "Empty text should return empty list"
#     print("✓ Empty text handled correctly")
    
#     # Single token text
#     chunks = tokenizer.tokenize_and_chunk_text("Hello", max_parent_chunk_size=100, max_child_chunk_size=25, parent_chunk_overlap_size=10, child_chunk_overlap_size=5)
#     assert len(chunks) == 1, "Single word should return one chunk"
#     assert isinstance(chunks[0], TextChunkDetails), "Chunk should be a TextChunkDetails instance"
#     assert len(chunks[0].embedding) > 0, "Chunk should have an embedding"
#     assert chunks[0].parent_text == chunks[0].text, "For single token, parent and child text should match"
#     print("✓ Single token text handled correctly")
#     print(f"  Embedding dimension: {len(chunks[0].embedding)}")
    
#     # max_parent_chunk_size == max_child_chunk_size (no subdivision)
#     chunks = tokenizer.tokenize_and_chunk_text("This is a test sentence for chunking.", max_parent_chunk_size=50, max_child_chunk_size=50, parent_chunk_overlap_size=0, child_chunk_overlap_size=0)
#     for chunk in chunks:
#         assert chunk.parent_text == chunk.text, "When sizes are equal, parent and child text should match"
#     print("✓ Equal parent/child chunk size handled correctly")
    
#     # Small child chunk size
#     chunks = tokenizer.tokenize_and_chunk_text("This is a test sentence.", max_parent_chunk_size=20, max_child_chunk_size=5, parent_chunk_overlap_size=2, child_chunk_overlap_size=1)
#     assert len(chunks) > 1, "Small child chunk size should create multiple chunks"
#     assert all(isinstance(chunk, TextChunkDetails) for chunk in chunks), "All chunks should be TextChunkDetails instances"
#     assert all(len(chunk.embedding) > 0 for chunk in chunks), "All chunks should have embeddings"
#     assert all(len(chunk.parent_text) >= len(chunk.text) for chunk in chunks), "Parent text should always be >= child text"
#     print(f"✓ Small child chunk size created {len(chunks)} child chunks")
    
#     # Validation errors
#     try:
#         tokenizer.tokenize_and_chunk_text("test", max_parent_chunk_size=0, max_child_chunk_size=25, parent_chunk_overlap_size=10, child_chunk_overlap_size=5)
#         assert False, "Should have raised ValueError"
#     except ValueError:
#         print("✓ max_parent_chunk_size=0 raises ValueError")
    
#     try:
#         tokenizer.tokenize_and_chunk_text("test", max_parent_chunk_size=100, max_child_chunk_size=0, parent_chunk_overlap_size=10, child_chunk_overlap_size=0)
#         assert False, "Should have raised ValueError"
#     except ValueError:
#         print("✓ max_child_chunk_size=0 raises ValueError")
    
#     try:
#         tokenizer.tokenize_and_chunk_text("test", max_parent_chunk_size=25, max_child_chunk_size=100, parent_chunk_overlap_size=10, child_chunk_overlap_size=5)
#         assert False, "Should have raised ValueError"
#     except ValueError:
#         print("✓ max_child_chunk_size > max_parent_chunk_size raises ValueError")
    
#     try:
#         tokenizer.tokenize_and_chunk_text("test", max_parent_chunk_size=100, max_child_chunk_size=25, parent_chunk_overlap_size=100, child_chunk_overlap_size=5)
#         assert False, "Should have raised ValueError"
#     except ValueError:
#         print("✓ parent_chunk_overlap_size >= max_parent_chunk_size raises ValueError")
    
#     try:
#         tokenizer.tokenize_and_chunk_text("test", max_parent_chunk_size=100, max_child_chunk_size=25, parent_chunk_overlap_size=10, child_chunk_overlap_size=25)
#         assert False, "Should have raised ValueError"
#     except ValueError:
#         print("✓ child_chunk_overlap_size >= max_child_chunk_size raises ValueError")
    
#     print("\nAll edge cases passed!")

# def test_text_chunk_details_model():
#     """Test the TextChunkDetails Pydantic model."""
#     print("\n" + "="*80)
#     print("Testing TextChunkDetails Model")
#     print("="*80)
    
#     # Create a sample chunk
#     chunk = TextChunkDetails(
#         text="Sample child text",
#         parent_text="Sample parent text with more context around the child text",
#         token_count=3,
#         start_char=0,
#         end_char=17,
#         embedding=[0.1, 0.2, 0.3]
#     )
    
#     print(f"✓ TextChunkDetails created successfully")
#     print(f"  Child text: {chunk.text}")
#     print(f"  Parent text: {chunk.parent_text}")
#     print(f"  Token count: {chunk.token_count}")
#     print(f"  Char range: {chunk.start_char}-{chunk.end_char}")
#     print(f"  Embedding: {chunk.embedding}")
    
#     # Test JSON serialization
#     chunk_json = chunk.model_dump_json()
#     print(f"✓ TextChunkDetails can be serialized to JSON")
#     print(f"  JSON length: {len(chunk_json)} characters")
    
#     # Test JSON deserialization
#     chunk_restored = TextChunkDetails.model_validate_json(chunk_json)
#     assert chunk_restored.text == chunk.text, "Restored chunk text should match original"
#     assert chunk_restored.parent_text == chunk.parent_text, "Restored parent_text should match original"
#     print(f"✓ TextChunkDetails can be deserialized from JSON")

# if __name__ == "__main__":
#     try:
#         test_basic_parent_child_chunking()
#         test_edge_cases()
#         test_text_chunk_details_model()
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()