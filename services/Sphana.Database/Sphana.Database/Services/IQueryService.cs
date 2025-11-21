using Sphana.Database.Models;

namespace Sphana.Database.Services;

/// <summary>
/// Interface for query service
/// </summary>
public interface IQueryService
{
    /// <summary>
    /// Executes a query using hybrid vector + graph retrieval
    /// </summary>
    /// <param name="query">The query text</param>
    /// <param name="tenantId">The tenant ID</param>
    /// <param name="indexName">The index name</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Query result with vector and graph results</returns>
    Task<QueryResult> ExecuteQueryAsync(
        string query,
        string tenantId,
        string indexName,
        CancellationToken cancellationToken = default);
}

