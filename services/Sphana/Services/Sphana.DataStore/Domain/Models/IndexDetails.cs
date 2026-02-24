namespace Sphana.DataStore.Domain.Models;

/// <summary>
/// Represents the metadata details of a vector index.
/// </summary>
public sealed class IndexDetails
{
    public required string IndexName { get; init; }
    public required string Description { get; init; }
    public required string MediaType { get; init; }
    public required int Dimension { get; init; }
    public required int NumberOfShards { get; init; }
    public required DateTime CreationTimestamp { get; init; }
    public required DateTime ModificationTimestamp { get; init; }
}