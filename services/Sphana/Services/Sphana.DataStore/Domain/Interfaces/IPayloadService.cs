namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Service interface for payload management operations.
/// </summary>
public interface IPayloadService
{
    void UploadPayload(string indexName, string entryName, byte[] payloadBytes);
    void AppendPayload(string indexName, string entryName, byte[] payloadBytes);
    void DeletePayload(string indexName, string entryName);
}