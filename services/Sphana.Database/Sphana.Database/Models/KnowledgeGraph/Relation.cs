namespace Sphana.Database.Models.KnowledgeGraph;

/// <summary>
/// Represents a relation (edge) in the knowledge graph
/// </summary>
public sealed class Relation
{
    /// <summary>
    /// Unique identifier for the relation
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
    /// Source entity ID (subject)
    /// </summary>
    public required string SourceEntityId { get; init; }

    /// <summary>
    /// Target entity ID (object)
    /// </summary>
    public required string TargetEntityId { get; init; }

    /// <summary>
    /// Relation type/predicate (e.g., "works_for", "located_in")
    /// </summary>
    public required string RelationType { get; init; }

    /// <summary>
    /// Confidence score of the relation extraction
    /// </summary>
    public float Confidence { get; init; }

    /// <summary>
    /// Source chunk ID where this relation was extracted
    /// </summary>
    public required string SourceChunkId { get; init; }

    /// <summary>
    /// Additional properties
    /// </summary>
    public Dictionary<string, string> Properties { get; init; } = new();

    /// <summary>
    /// Timestamp when the relation was created
    /// </summary>
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
}

