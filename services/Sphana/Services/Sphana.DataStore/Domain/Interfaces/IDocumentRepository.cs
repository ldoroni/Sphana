using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Generic interface for RocksDB-backed key-value document storage.
/// </summary>
public interface IDocumentRepository<TDocument> where TDocument : class
{
    /// <summary>
    /// Ensure the table (column family) exists for the given table name.
    /// </summary>
    void InitializeTable(string tableName);

    /// <summary>
    /// Drop the table and all its data from disk.
    /// </summary>
    void DropTable(string tableName);

    /// <summary>
    /// Insert or update a document identified by the given key.
    /// </summary>
    void Upsert(string tableName, string documentId, TDocument document);

    /// <summary>
    /// Delete a document by its key.
    /// </summary>
    void Delete(string tableName, string documentId);

    /// <summary>
    /// Read a document by its key. Returns null if not found.
    /// </summary>
    TDocument? Read(string tableName, string documentId);

    /// <summary>
    /// Check whether a document with the given key exists.
    /// </summary>
    bool Exists(string tableName, string documentId);

    /// <summary>
    /// List all keys stored in the given table.
    /// </summary>
    IReadOnlyList<string> ListKeys(string tableName);
}