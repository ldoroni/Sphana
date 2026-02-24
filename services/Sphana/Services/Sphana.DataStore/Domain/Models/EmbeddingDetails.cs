namespace Sphana.DataStore.Domain.Models;

/// <summary>
/// Represents embedding details for an entry within an index.
/// </summary>
public sealed class EmbeddingDetails
{
    public required string EntryName { get; init; }
    public required string ShardName { get; init; }
    public required long VectorOffset { get; init; }
}