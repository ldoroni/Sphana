from injector import singleton
from sphana_rag.models.tokenized_text import TokenizedText

@singleton
class TokenChunkerService:

    def __init__(self):
        pass

    def chunk_tokens(self, tokenized_text: TokenizedText, max_chunk_size: int, chunk_overlap_size: int) -> list[TokenizedText]:
        if not tokenized_text.token_ids:
            return []

        if max_chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {max_chunk_size}")
        if chunk_overlap_size < 0:
            raise ValueError(f"overlap_size must be non-negative, got {chunk_overlap_size}")
        if chunk_overlap_size >= max_chunk_size:
            raise ValueError(f"overlap_size ({chunk_overlap_size}) must be less than chunk_size ({max_chunk_size})")

        total_tokens: int = len(tokenized_text.token_ids)
        step: int = max_chunk_size - chunk_overlap_size
        chunks: list[TokenizedText] = []

        pos: int = 0
        while pos < total_tokens:
            end: int = min(pos + max_chunk_size, total_tokens)

            chunk_token_ids: list[int] = tokenized_text.token_ids[pos:end]
            chunk_offsets: list[tuple[int, int]] = tokenized_text.offsets[pos:end]

            # Determine character span from offset mappings
            start_char: int = chunk_offsets[0][0]
            end_char: int = chunk_offsets[-1][1]
            chunk_text: str = tokenized_text.text[start_char:end_char]

            chunks.append(TokenizedText(
                text=chunk_text,
                token_ids=chunk_token_ids,
                offsets=chunk_offsets,
                # token_count=len(chunk_token_ids),
                # start_char=start_char,
                # end_char=end_char
            ))

            # If we've reached the end, stop
            if end >= total_tokens:
                break

            pos += step

        return chunks