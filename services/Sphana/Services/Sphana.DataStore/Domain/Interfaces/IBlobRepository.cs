namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Repository for binary blob storage backed by LMDB.
/// </summary>
public interface IBlobRepository
{
    void InitializeStorage(string storageName);
    void DropStorage(string storageName);
    void WriteBlob(string storageName, string blobId, byte[] buffer);
    void AppendBlob(string storageName, string blobId, byte[] buffer);
    byte[]? ReadBlob(string storageName, string blobId);
    byte[]? ReadBlobChunk(string storageName, string blobId, int startIndex, int endIndex);
    void DeleteBlob(string storageName, string blobId);
    bool BlobExists(string storageName, string blobId);
}