using LightningDB;
using Microsoft.Extensions.Logging;
using Sphana.DataStore.Domain.Interfaces;

namespace Sphana.DataStore.Infrastructure.Persistence.Lmdb;

/// <summary>
/// LMDB-backed blob repository replacing the Python Zarr-based blob storage.
/// Uses Lightning.NET for cross-platform LMDB access. Each storageName maps to a named LMDB database.
/// </summary>
public sealed class LmdbBlobRepository : IBlobRepository, IDisposable
{
    private readonly LightningEnvironment _environment;
    private readonly ILogger<LmdbBlobRepository> _logger;
    private readonly HashSet<string> _initializedStorages = new();
    private bool _disposed;

    public LmdbBlobRepository(string databasePath, ILogger<LmdbBlobRepository> logger)
    {
        _logger = logger;
        Directory.CreateDirectory(databasePath);

        _environment = new LightningEnvironment(databasePath)
        {
            MaxDatabases = 256,
            MapSize = 10L * 1024 * 1024 * 1024 // 10 GB
        };
        _environment.Open();

        _logger.LogInformation("Opened LMDB environment at '{DatabasePath}'", databasePath);
    }

    public void InitializeStorage(string storageName)
    {
        // Create the named database by opening it with Create flag in a write transaction
        using var transaction = _environment.BeginTransaction();
        using var database = transaction.OpenDatabase(storageName, new DatabaseConfiguration { Flags = DatabaseOpenFlags.Create });
        transaction.Commit();
        _initializedStorages.Add(storageName);
        _logger.LogInformation("Initialized LMDB storage '{StorageName}'", storageName);
    }

    public void DropStorage(string storageName)
    {
        using var transaction = _environment.BeginTransaction();
        using var database = transaction.OpenDatabase(storageName);
        transaction.DropDatabase(database);
        transaction.Commit();
        _initializedStorages.Remove(storageName);
        _logger.LogInformation("Dropped LMDB storage '{StorageName}'", storageName);
    }

    public void WriteBlob(string storageName, string blobId, byte[] buffer)
    {
        using var transaction = _environment.BeginTransaction();
        using var database = transaction.OpenDatabase(storageName);
        transaction.Put(database, System.Text.Encoding.UTF8.GetBytes(blobId), buffer);
        transaction.Commit();
    }

    public void AppendBlob(string storageName, string blobId, byte[] buffer)
    {
        var existing = ReadBlob(storageName, blobId);
        if (existing is null)
        {
            WriteBlob(storageName, blobId, buffer);
            return;
        }

        var combined = new byte[existing.Length + buffer.Length];
        Buffer.BlockCopy(existing, 0, combined, 0, existing.Length);
        Buffer.BlockCopy(buffer, 0, combined, existing.Length, buffer.Length);
        WriteBlob(storageName, blobId, combined);
    }

    public byte[]? ReadBlob(string storageName, string blobId)
    {
        using var transaction = _environment.BeginTransaction(TransactionBeginFlags.ReadOnly);
        using var database = transaction.OpenDatabase(storageName);
        var (resultCode, _, valueBuffer) = transaction.Get(database, System.Text.Encoding.UTF8.GetBytes(blobId));

        if (resultCode == MDBResultCode.NotFound)
            return null;

        return valueBuffer.CopyToNewArray();
    }

    public byte[]? ReadBlobChunk(string storageName, string blobId, int startIndex, int endIndex)
    {
        var fullBlob = ReadBlob(storageName, blobId);
        if (fullBlob is null)
            return null;

        var length = Math.Min(endIndex, fullBlob.Length) - startIndex;
        if (length <= 0 || startIndex >= fullBlob.Length)
            return null;

        var chunk = new byte[length];
        Buffer.BlockCopy(fullBlob, startIndex, chunk, 0, length);
        return chunk;
    }

    public void DeleteBlob(string storageName, string blobId)
    {
        using var transaction = _environment.BeginTransaction();
        using var database = transaction.OpenDatabase(storageName);
        transaction.Delete(database, System.Text.Encoding.UTF8.GetBytes(blobId));
        transaction.Commit();
    }

    public bool BlobExists(string storageName, string blobId)
    {
        return ReadBlob(storageName, blobId) is not null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _environment.Dispose();
        GC.SuppressFinalize(this);
    }
}