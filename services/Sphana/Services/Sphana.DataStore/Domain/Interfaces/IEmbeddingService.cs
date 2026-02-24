namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Service interface for embedding management operations.
/// </summary>
public interface IEmbeddingService
{
    int AddEmbeddings(string indexName, IReadOnlyList<EmbeddingEntry> entries);
    int ResetEmbeddings(string indexName);
}

/// <summary>
/// Represents a single entry with its embeddings for the AddEmbeddings operation.
/// </summary>
public sealed class EmbeddingEntry
{
    public required string EntryName { get; init; }
    public required IReadOnlyList<float[]> Embeddings { get; init; }
}