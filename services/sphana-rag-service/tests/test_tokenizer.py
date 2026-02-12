# """
# Test script for TextTokenizer, TokenChunker, and TokenEmbedder classes.
# Run this after installing dependencies with: pip install -e .
# """

# from sphana_rag.services.tokenizer import TextTokenizer, TokenChunker, TokenEmbedder
# from sphana_rag.models import TokenizedText, TokenChunk

# def test_tokenizer():
#     """Test TextTokenizer produces token IDs and offsets."""
#     print("=" * 80)
#     print("Testing TextTokenizer")
#     print("=" * 80)
    
#     tokenizer = TextTokenizer()
    
#     sample_text = "Artificial intelligence is intelligence demonstrated by machines."
#     result: TokenizedText = tokenizer.tokenize(sample_text)
    
#     print(f"Text: {sample_text}")
#     print(f"Token IDs ({len(result.token_ids)}): {result.token_ids}")
#     print(f"Offsets ({len(result.offsets)}): {result.offsets}")
    
#     assert len(result.token_ids) > 0, "Should produce tokens"
#     assert len(result.token_ids) == len(result.offsets), "Token IDs and offsets should have same length"
#     print("✓ TextTokenizer works correctly")

#     # Empty text
#     empty_result = tokenizer.tokenize("")
#     assert len(empty_result.token_ids) == 0, "Empty text should produce no tokens"
#     print("✓ Empty text handled correctly")

# def test_token_chunker():
#     """Test TokenChunker produces correct chunks from token arrays."""
#     print("\n" + "=" * 80)
#     print("Testing TokenChunker")
#     print("=" * 80)
    
#     tokenizer = TextTokenizer()
    
#     sample_text = """
#     Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural 
#     intelligence displayed by humans and animals. Leading AI textbooks define the field as the study 
#     of "intelligent agents": any device that perceives its environment and takes actions that maximize 
#     its chance of successfully achieving its goals.
#     """
    
#     # Step 1: Tokenize
#     tokenized: TokenizedText = tokenizer.tokenize(sample_text)
#     print(f"Total tokens: {len(tokenized.token_ids)}")
    
#     # Step 2: Chunk into parent chunks
#     parent_chunks: list[TokenChunk] = TokenChunker.chunk_tokens(
#         token_ids=tokenized.token_ids,
#         offsets=tokenized.offsets,
#         text=sample_text,
#         chunk_size=50,
#         overlap_size=5
#     )
    
#     print(f"\nParent chunks ({len(parent_chunks)}):")
#     for i, chunk in enumerate(parent_chunks):
#         print(f"  Parent {i}: {chunk.token_count} tokens, chars [{chunk.start_char}:{chunk.end_char}]")
#         print(f"    Text: {chunk.text[:80]}..." if len(chunk.text) > 80 else f"    Text: {chunk.text}")
    
#     assert len(parent_chunks) > 0, "Should produce parent chunks"
    
#     # Step 3: Chunk each parent into child chunks
#     for i, parent_chunk in enumerate(parent_chunks):
#         child_chunks: list[TokenChunk] = TokenChunker.chunk_tokens(
#             token_ids=parent_chunk.token_ids,
#             offsets=parent_chunk.offsets,
#             text=parent_chunk.text,
#             chunk_size=15,
#             overlap_size=3
#         )
#         print(f"\n  Parent {i} -> {len(child_chunks)} child chunks:")
#         for j, child in enumerate(child_chunks):
#             print(f"    Child {j}: {child.token_count} tokens, text: {child.text[:60]}...")
    
#     print("✓ TokenChunker parent-child chunking works correctly")

#     # Edge case: empty tokens
#     empty_chunks = TokenChunker.chunk_tokens([], [], "", chunk_size=10, overlap_size=2)
#     assert len(empty_chunks) == 0, "Empty tokens should produce no chunks"
#     print("✓ Empty tokens handled correctly")

#     # Validation errors
#     try:
#         TokenChunker.chunk_tokens([1], [(0, 1)], "a", chunk_size=0, overlap_size=0)
#         assert False, "Should have raised ValueError"
#     except ValueError:
#         print("✓ chunk_size=0 raises ValueError")
    
#     try:
#         TokenChunker.chunk_tokens([1], [(0, 1)], "a", chunk_size=5, overlap_size=5)
#         assert False, "Should have raised ValueError"
#     except ValueError:
#         print("✓ overlap_size >= chunk_size raises ValueError")

# def test_token_embedder():
#     """Test TokenEmbedder produces embeddings."""
#     print("\n" + "=" * 80)
#     print("Testing TokenEmbedder")
#     print("=" * 80)
    
#     embedder = TokenEmbedder()
    
#     # Single text
#     embedding = embedder.embed_text("search_query: What is artificial intelligence?")
#     print(f"Single embedding dimension: {len(embedding)}")
#     print(f"Sample (first 5): {embedding[:5]}")
#     assert len(embedding) > 0, "Should produce an embedding"
#     print("✓ Single text embedding works")
    
#     # Batch texts
#     texts = [
#         "search_document: AI is demonstrated by machines.",
#         "search_document: Natural language processing is a subfield of AI.",
#         "search_document: Machine learning uses statistical methods."
#     ]
#     embeddings = embedder.embed_texts(texts)
#     print(f"Batch embeddings: {len(embeddings)} x {len(embeddings[0])}")
#     assert len(embeddings) == 3, "Should produce 3 embeddings"
#     assert all(len(e) == len(embedding) for e in embeddings), "All embeddings should have same dimension"
#     print("✓ Batch text embedding works")
    
#     # Empty batch
#     empty_embeddings = embedder.embed_texts([])
#     assert len(empty_embeddings) == 0, "Empty batch should return empty list"
#     print("✓ Empty batch handled correctly")

# def test_full_pipeline():
#     """Test the full tokenize -> chunk -> embed pipeline."""
#     print("\n" + "=" * 80)
#     print("Testing Full Pipeline: Tokenize -> Chunk -> Embed")
#     print("=" * 80)
    
#     tokenizer = TextTokenizer()
#     embedder = TokenEmbedder()
    
#     sample_text = """
#     Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural 
#     intelligence displayed by humans and animals. Leading AI textbooks define the field as the study 
#     of "intelligent agents": any device that perceives its environment and takes actions that maximize 
#     its chance of successfully achieving its goals.
#     """
    
#     # Step 1: Tokenize
#     tokenized = tokenizer.tokenize(sample_text)
#     print(f"1. Tokenized: {len(tokenized.token_ids)} tokens")
    
#     # Step 2: Parent chunks
#     parent_chunks = TokenChunker.chunk_tokens(
#         token_ids=tokenized.token_ids,
#         offsets=tokenized.offsets,
#         text=sample_text,
#         chunk_size=50,
#         overlap_size=5
#     )
#     print(f"2. Parent chunks: {len(parent_chunks)}")
    
#     # Step 3: Child chunks from each parent
#     all_child_texts = []
#     parent_child_map = []
#     for pi, parent in enumerate(parent_chunks):
#         children = TokenChunker.chunk_tokens(
#             token_ids=parent.token_ids,
#             offsets=parent.offsets,
#             text=parent.text,
#             chunk_size=15,
#             overlap_size=3
#         )
#         for child in children:
#             parent_child_map.append((pi, child))
#             all_child_texts.append(f"search_document: {child.text}")
    
#     print(f"3. Child chunks: {len(parent_child_map)}")
    
#     # Step 4: Batch embed all child texts
#     embeddings = embedder.embed_texts(all_child_texts)
#     print(f"4. Embeddings: {len(embeddings)} x {len(embeddings[0])}")
    
#     assert len(embeddings) == len(parent_child_map), "Should have one embedding per child chunk"
    
#     # Display summary
#     for i, (pi, child) in enumerate(parent_child_map):
#         print(f"  Child {i} (parent {pi}): {child.token_count} tokens, embedding dim={len(embeddings[i])}")
#         print(f"    Text: {child.text[:60]}...")
    
#     print("\n✓ Full pipeline works correctly!")

# if __name__ == "__main__":
#     try:
#         test_tokenizer()
#         test_token_chunker()
#         test_token_embedder()
#         test_full_pipeline()
#         print("\n" + "=" * 80)
#         print("ALL TESTS PASSED!")
#         print("=" * 80)
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()