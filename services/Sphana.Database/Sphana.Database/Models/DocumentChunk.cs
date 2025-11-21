namespace Sphana.Database.Models;

/// <summary>
/// Represents a chunk of a document created during semantic chunking
/// </summary>
public sealed class DocumentChunk
{
    /// <summary>
    /// Unique identifier for the chunk
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Parent document ID
    /// </summary>
    public required string DocumentId { get; init; }

    /// <summary>
    /// Tenant identifier
    /// </summary>
    public required string TenantId { get; init; }

    /// <summary>
    /// Index name
    /// </summary>
    public required string IndexName { get; init; }

    /// <summary>
    /// Chunk text content
    /// </summary>
    public required string Content { get; init; }

    /// <summary>
    /// Chunk index in the document
    /// </summary>
    public int ChunkIndex { get; init; }

    /// <summary>
    /// Start position in the original document
    /// </summary>
    public int StartPosition { get; init; }

    /// <summary>
    /// End position in the original document
    /// </summary>
    public int EndPosition { get; init; }

    /// <summary>
    /// Dense embedding vector (normalized, quantized to int8 for storage)
    /// </summary>
    public float[]? Embedding { get; set; }

    /// <summary>
    /// Quantized embedding (int8) for efficient storage
    /// </summary>
    public byte[]? QuantizedEmbedding { get; set; }

    /// <summary>
    /// Embedding dimension
    /// </summary>
    public int EmbeddingDimension { get; init; }

    /// <summary>
    /// Timestamp when the chunk was created
    /// </summary>
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
}

