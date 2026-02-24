using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Service interface for index management operations.
/// </summary>
public interface IIndexService
{
    IndexDetails CreateIndex(string indexName, string description, string mediaType, int dimension, int numberOfShards);
    IndexDetails GetIndex(string indexName);
    IReadOnlyList<IndexDetails> ListIndices();
    IndexDetails UpdateIndex(string indexName, string description);
    void DeleteIndex(string indexName);
    bool IndexExists(string indexName);
}
