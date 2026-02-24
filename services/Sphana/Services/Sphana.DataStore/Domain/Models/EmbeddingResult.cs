namespace Sphana.DataStore.Domain.Models;

/// <summary>
/// Represents a single embedding result from a vector query.
/// </summary>
public sealed class EmbeddingResult
{
    public required string EntryName { get; init; }
    public required float Distance { get; init; }
}