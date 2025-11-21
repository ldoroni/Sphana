namespace Sphana.Database.Models.KnowledgeGraph;

/// <summary>
/// Represents a subgraph extracted from the main knowledge graph for reasoning
/// </summary>
public sealed class KnowledgeSubgraph
{
    /// <summary>
    /// Unique identifier for the subgraph
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Entities in this subgraph
    /// </summary>
    public required IReadOnlyList<Entity> Entities { get; init; }

    /// <summary>
    /// Relations in this subgraph
    /// </summary>
    public required IReadOnlyList<Relation> Relations { get; init; }

    /// <summary>
    /// Relevance score assigned by the GNN ranker
    /// </summary>
    public float RelevanceScore { get; set; }

    /// <summary>
    /// Query that generated this subgraph
    /// </summary>
    public string? SourceQuery { get; init; }

    /// <summary>
    /// Reasoning path from question entities to answer entities
    /// </summary>
    public List<string> ReasoningPath { get; init; } = new();
}

