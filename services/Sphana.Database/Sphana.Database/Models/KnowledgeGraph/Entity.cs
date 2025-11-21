namespace Sphana.Database.Models.KnowledgeGraph;

/// <summary>
/// Represents an entity in the knowledge graph
/// </summary>
public sealed class Entity
{
    /// <summary>
    /// Unique identifier for the entity (node ID in the graph)
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Tenant identifier
    /// </summary>
    public required string TenantId { get; init; }

    /// <summary>
    /// Index name
    /// </summary>
    public required string IndexName { get; init; }

    /// <summary>
    /// Entity text/name
    /// </summary>
    public required string Text { get; init; }

    /// <summary>
    /// Entity type (e.g., PERSON, ORGANIZATION, LOCATION)
    /// </summary>
    public required string Type { get; init; }

    /// <summary>
    /// Source chunk ID where this entity was extracted
    /// </summary>
    public required string SourceChunkId { get; init; }

    /// <summary>
    /// Entity embedding vector (for semantic similarity)
    /// </summary>
    public float[]? Embedding { get; set; }

    /// <summary>
    /// Quantized embedding
    /// </summary>
    public byte[]? QuantizedEmbedding { get; set; }

    /// <summary>
    /// Additional properties
    /// </summary>
    public Dictionary<string, string> Properties { get; init; } = new();

    /// <summary>
    /// Timestamp when the entity was created
    /// </summary>
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
}

