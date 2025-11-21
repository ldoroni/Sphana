using System.Collections.Concurrent;
using System.Numerics;

namespace Sphana.Database.Infrastructure.VectorIndex;

/// <summary>
/// Interface for vector index operations
/// </summary>
public interface IVectorIndex
{
    Task AddAsync(string id, float[] vector, CancellationToken cancellationToken = default);
    Task AddBatchAsync(IEnumerable<(string Id, float[] Vector)> items, CancellationToken cancellationToken = default);
    Task<List<SearchResult>> SearchAsync(float[] queryVector, int topK, CancellationToken cancellationToken = default);
    Task<bool> RemoveAsync(string id, CancellationToken cancellationToken = default);
    Task SaveAsync(string path, CancellationToken cancellationToken = default);
    Task LoadAsync(string path, CancellationToken cancellationToken = default);
    int Count { get; }
}

/// <summary>
/// Search result from vector index
/// </summary>
public sealed class SearchResult
{
    public required string Id { get; init; }
    public required float Score { get; init; }
    public float[]? Vector { get; init; }
}

