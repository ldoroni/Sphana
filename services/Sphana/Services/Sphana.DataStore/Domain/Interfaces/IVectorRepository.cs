using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Repository for FAISS vector index operations.
/// </summary>
public interface IVectorRepository
{
    void InitializeIndex(string indexName, int dimension);
    void DropIndex(string indexName);
    void AddEmbedding(string indexName, string embeddingId, float[] embedding);
    void DeleteEmbedding(string indexName, string embeddingId);
    IReadOnlyList<EmbeddingResult> Search(string indexName, float[] queryVector, int maxResults);
    int Count(string indexName);
}