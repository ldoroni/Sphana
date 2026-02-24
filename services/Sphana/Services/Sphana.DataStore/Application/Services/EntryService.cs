using Microsoft.Extensions.Logging;
using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Infrastructure.Configuration;
using Sphana.ManagedException.Arguments;

namespace Sphana.DataStore.Application.Services;

/// <summary>
/// Manages entry listing and deletion across index shards.
/// </summary>
public sealed class EntryService : IEntryService
{
    private readonly IIndexDetailsRepository _indexDetailsRepository;
    private readonly IEmbeddingDetailsRepository _embeddingDetailsRepository;
    private readonly IVectorRepository _vectorRepository;
    private readonly IBlobRepository _blobRepository;
    private readonly DataStoreConfiguration _configuration;
    private readonly ILogger<EntryService> _logger;

    public EntryService(
        IIndexDetailsRepository indexDetailsRepository,
        IEmbeddingDetailsRepository embeddingDetailsRepository,
        IVectorRepository vectorRepository,
        IBlobRepository blobRepository,
        DataStoreConfiguration configuration,
        ILogger<EntryService> logger)
    {
        _indexDetailsRepository = indexDetailsRepository;
        _embeddingDetailsRepository = embeddingDetailsRepository;
        _vectorRepository = vectorRepository;
        _blobRepository = blobRepository;
        _configuration = configuration;
        _logger = logger;
    }

    public IReadOnlyList<string> ListEntries(string indexName)
    {
        _logger.LogInformation("Listing entries for index '{IndexName}'", indexName);

        var indexDetails = _indexDetailsRepository.Read(indexName);
        if (indexDetails is null)
            throw new ItemNotFoundException($"Index '{indexName}' not found.");

        var allEntryNames = new HashSet<string>();

        for (var shardIndex = 0; shardIndex < indexDetails.NumberOfShards; shardIndex++)
        {
            var shardName = BuildShardName(indexName, shardIndex);
            var entryNames = _embeddingDetailsRepository.ListEntryNames(shardName);

            foreach (var entryName in entryNames)
            {
                allEntryNames.Add(entryName);
            }
        }

        _logger.LogInformation("Found {Count} entries in index '{IndexName}'", allEntryNames.Count, indexName);
        return allEntryNames.ToList().AsReadOnly();
    }

    public void DeleteEntry(string indexName, string entryName)
    {
        _logger.LogInformation("Deleting entry '{EntryName}' from index '{IndexName}'", entryName, indexName);

        var indexDetails = _indexDetailsRepository.Read(indexName);
        if (indexDetails is null)
            throw new ItemNotFoundException($"Index '{indexName}' not found.");

        for (var shardIndex = 0; shardIndex < indexDetails.NumberOfShards; shardIndex++)
        {
            var shardName = BuildShardName(indexName, shardIndex);
            var embeddingDetails = _embeddingDetailsRepository.Read(shardName, entryName);

            if (embeddingDetails is not null)
            {
                _vectorRepository.DeleteEmbedding(shardName, entryName);
                _embeddingDetailsRepository.Delete(shardName, entryName);
            }
        }

        _blobRepository.DeleteBlob(indexName, entryName);

        _logger.LogInformation("Entry '{EntryName}' deleted from index '{IndexName}'", entryName, indexName);
    }

    private string BuildShardName(string indexName, int shardIndex)
    {
        var shardSuffix = shardIndex == 0
            ? _configuration.DefaultShardName
            : $"shard_{shardIndex}";
        return $"{indexName}:{shardSuffix}";
    }
}