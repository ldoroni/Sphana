namespace Sphana.DataStore.Domain.Models;

/// <summary>
/// Represents shard metadata for an index.
/// </summary>
public sealed class ShardDetails
{
    public required string ShardName { get; init; }
    public required string IndexName { get; init; }
    public required int VectorCount { get; init; }
}