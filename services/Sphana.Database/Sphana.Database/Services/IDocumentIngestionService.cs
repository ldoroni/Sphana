using Sphana.Database.Models;

namespace Sphana.Database.Services;

/// <summary>
/// Interface for document ingestion service
/// </summary>
public interface IDocumentIngestionService
{
    /// <summary>
    /// Ingests a document into the system, performing chunking, embedding generation,
    /// relation extraction, and indexing
    /// </summary>
    /// <param name="document">The document to ingest</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>The document ID</returns>
    Task<string> IngestDocumentAsync(Document document, CancellationToken cancellationToken = default);
}

