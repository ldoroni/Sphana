using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Service interface for query execution operations.
/// </summary>
public interface IQueryService
{
    IReadOnlyList<ExecuteQueryResult> ExecuteQuery(string indexName, float[] embeddings, int limit);
}