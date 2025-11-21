namespace Sphana.Database.Models;

/// <summary>
/// Represents a document in the NRDB system
/// </summary>
public sealed class Document
{
    /// <summary>
    /// Unique identifier for the document
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Tenant identifier for multi-tenancy
    /// </summary>
    public required string TenantId { get; init; }

    /// <summary>
    /// Index name where the document belongs
    /// </summary>
    public required string IndexName { get; init; }

    /// <summary>
    /// Document title
    /// </summary>
    public required string Title { get; init; }

    /// <summary>
    /// Document content
    /// </summary>
    public required string Content { get; init; }

    /// <summary>
    /// Additional metadata
    /// </summary>
    public Dictionary<string, string> Metadata { get; init; } = new();

    /// <summary>
    /// Timestamp when the document was indexed
    /// </summary>
    public DateTime IndexedAt { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Hash of the document content for deduplication
    /// </summary>
    public string? ContentHash { get; init; }
}

