using Microsoft.Extensions.Logging;
using Sphana.DataStore.Domain.Interfaces;
using Sphana.ManagedException.Arguments;

namespace Sphana.DataStore.Application.Services;

/// <summary>
/// Manages payload blob storage operations: upload, append, delete.
/// </summary>
public sealed class PayloadService : IPayloadService
{
    private readonly IIndexDetailsRepository _indexDetailsRepository;
    private readonly IBlobRepository _blobRepository;
    private readonly ILogger<PayloadService> _logger;

    public PayloadService(
        IIndexDetailsRepository indexDetailsRepository,
        IBlobRepository blobRepository,
        ILogger<PayloadService> logger)
    {
        _indexDetailsRepository = indexDetailsRepository;
        _blobRepository = blobRepository;
        _logger = logger;
    }

    public void UploadPayload(string indexName, string entryName, byte[] payloadBytes)
    {
        _logger.LogInformation("Uploading payload for entry '{EntryName}' in index '{IndexName}'",
            entryName, indexName);

        ValidateIndexExists(indexName);
        _blobRepository.WriteBlob(indexName, entryName, payloadBytes);

        _logger.LogInformation("Payload uploaded for entry '{EntryName}' in index '{IndexName}'",
            entryName, indexName);
    }

    public void AppendPayload(string indexName, string entryName, byte[] payloadBytes)
    {
        _logger.LogInformation("Appending payload for entry '{EntryName}' in index '{IndexName}'",
            entryName, indexName);

        ValidateIndexExists(indexName);
        _blobRepository.AppendBlob(indexName, entryName, payloadBytes);

        _logger.LogInformation("Payload appended for entry '{EntryName}' in index '{IndexName}'",
            entryName, indexName);
    }

    public void DeletePayload(string indexName, string entryName)
    {
        _logger.LogInformation("Deleting payload for entry '{EntryName}' in index '{IndexName}'",
            entryName, indexName);

        ValidateIndexExists(indexName);
        _blobRepository.DeleteBlob(indexName, entryName);

        _logger.LogInformation("Payload deleted for entry '{EntryName}' in index '{IndexName}'",
            entryName, indexName);
    }

    private void ValidateIndexExists(string indexName)
    {
        if (!_indexDetailsRepository.Exists(indexName))
            throw new ItemNotFoundException($"Index '{indexName}' not found.");
    }
}