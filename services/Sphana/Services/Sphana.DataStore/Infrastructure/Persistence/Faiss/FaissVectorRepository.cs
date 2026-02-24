using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Domain.Models;
using Sphana.ManagedException.Internal;

namespace Sphana.DataStore.Infrastructure.Persistence.Faiss;

/// <summary>
/// FAISS-backed vector repository using P/Invoke into the native FAISS C library.
/// Manages multiple named indices keyed by index name.
/// </summary>
public sealed class FaissVectorRepository : IVectorRepository, IDisposable
{
    private readonly ConcurrentDictionary<string, nint> _indexPointers = new();
    private readonly ConcurrentDictionary<string, int> _dimensions = new();
    private readonly ILogger<FaissVectorRepository> _logger;
    private bool _disposed;

    public FaissVectorRepository(ILogger<FaissVectorRepository> logger)
    {
        _logger = logger;
    }

    public void InitializeIndex(string indexName, int dimension)
    {
        if (_indexPointers.ContainsKey(indexName))
        {
            _logger.LogWarning("FAISS index '{IndexName}' already initialized, skipping", indexName);
            return;
        }

        var resultCode = FaissNativeMethods.IndexFlatL2NewWith(out var indexPointer, dimension);
        if (resultCode != 0)
            throw new InternalErrorException($"Failed to create FAISS index '{indexName}' with dimension {dimension}. Error code: {resultCode}");

        _indexPointers[indexName] = indexPointer;
        _dimensions[indexName] = dimension;
        _logger.LogInformation("Initialized FAISS FlatL2 index '{IndexName}' with dimension {Dimension}", indexName, dimension);
    }

    public void DropIndex(string indexName)
    {
        if (_indexPointers.TryRemove(indexName, out var indexPointer))
        {
            FaissNativeMethods.IndexFree(indexPointer);
            _dimensions.TryRemove(indexName, out _);
            _logger.LogInformation("Dropped FAISS index '{IndexName}'", indexName);
        }
    }

    public void AddEmbedding(string indexName, string embeddingId, float[] embedding)
    {
        var indexPointer = GetIndexPointer(indexName);
        var resultCode = FaissNativeMethods.IndexAdd(indexPointer, 1, embedding);
        if (resultCode != 0)
            throw new InternalErrorException($"Failed to add embedding '{embeddingId}' to FAISS index '{indexName}'. Error code: {resultCode}");
    }

    public void DeleteEmbedding(string indexName, string embeddingId)
    {
        // FlatL2 index does not support individual deletion.
        // This is a known limitation matching the Python implementation which rebuilds the index on reset.
        _logger.LogWarning("Individual embedding deletion is not supported by FAISS FlatL2. Use reset instead. Index: '{IndexName}', EmbeddingId: '{EmbeddingId}'", indexName, embeddingId);
    }

    public IReadOnlyList<EmbeddingResult> Search(string indexName, float[] queryVector, int maxResults)
    {
        var indexPointer = GetIndexPointer(indexName);
        var totalVectors = FaissNativeMethods.IndexNTotal(indexPointer);

        if (totalVectors == 0)
            return Array.Empty<EmbeddingResult>();

        var effectiveMaxResults = (int)Math.Min(maxResults, totalVectors);
        var distances = new float[effectiveMaxResults];
        var labels = new long[effectiveMaxResults];

        var resultCode = FaissNativeMethods.IndexSearch(indexPointer, 1, queryVector, effectiveMaxResults, distances, labels);
        if (resultCode != 0)
            throw new InternalErrorException($"FAISS search failed on index '{indexName}'. Error code: {resultCode}");

        var results = new List<EmbeddingResult>(effectiveMaxResults);
        for (var resultIndex = 0; resultIndex < effectiveMaxResults; resultIndex++)
        {
            if (labels[resultIndex] < 0)
                continue;

            results.Add(new EmbeddingResult
            {
                EntryName = labels[resultIndex].ToString(),
                Distance = distances[resultIndex]
            });
        }

        return results;
    }

    public int Count(string indexName)
    {
        var indexPointer = GetIndexPointer(indexName);
        return (int)FaissNativeMethods.IndexNTotal(indexPointer);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var kvp in _indexPointers)
        {
            FaissNativeMethods.IndexFree(kvp.Value);
        }
        _indexPointers.Clear();
        _dimensions.Clear();

        GC.SuppressFinalize(this);
    }

    private nint GetIndexPointer(string indexName)
    {
        if (!_indexPointers.TryGetValue(indexName, out var indexPointer))
            throw new InternalErrorException($"FAISS index '{indexName}' is not initialized.");

        return indexPointer;
    }
}