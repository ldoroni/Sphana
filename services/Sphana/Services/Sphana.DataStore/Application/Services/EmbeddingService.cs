using Microsoft.Extensions.Logging;
using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Domain.Models;
using Sphana.DataStore.Infrastructure.Configuration;
using Sphana.ManagedException.Arguments;

namespace Sphana.DataStore.Application.Services;

/// <summary>
/// Manages adding and resetting vector embeddings across index shards.
/// </summary>
public sealed class EmbeddingService : IEmbeddingService
{
    private readonly IIndexDetailsRepository _indexDetailsRepository;
    private readonly IShardDetailsRepository _shardDetailsRepository;
    private readonly IEmbeddingDetailsRepository _embeddingDetailsRepository;
    private readonly IVectorRepository _vectorRepository;
    private readonly DataStoreConfiguration _configuration;
    private readonly ILogger<EmbeddingService> _logger;

    public EmbeddingService(
        IIndexDetailsRepository indexDetailsRepository,
        IShardDetailsRepository shardDetailsRepository,
        IEmbeddingDetailsRepository embeddingDetailsRepository,
        IVectorRepository vectorRepository,
        DataStoreConfiguration configuration,
        ILogger<EmbeddingService> logger)
    {
        _indexDetailsRepository = indexDetailsRepository;
        _shardDetailsRepository = shardDetailsRepository;
        _embeddingDetailsRepository = embeddingDetailsRepository;
        _vectorRepository = vectorRepository;
        _configuration = configuration;
        _logger = logger;
    }

    public int AddEmbeddings(string indexName, IReadOnlyList<EmbeddingEntry> entries)
    {
        _logger.LogInformation("Adding embeddings to index '{IndexName}' for {EntryCount} entries",
            indexName, entries.Count);

        var indexDetails = _indexDetailsRepository.Read(indexName);
        if (indexDetails is null)
            throw new ItemNotFoundException($"Index '{indexName}' not found.");

        var totalEmbeddingsStored = 0;

        foreach (var entry in entries)
        {
            for (var embeddingIndex = 0; embeddingIndex < entry.Embeddings.Count; embeddingIndex++)
            {
                var shardIndex = totalEmbeddingsStored % indexDetails.NumberOfShards;
                var shardName = BuildShardName(indexName, shardIndex);
                var embeddingId = $"{entry.EntryName}:{embeddingIndex}";

                _vectorRepository.AddEmbedding(shardName, embeddingId, entry.Embeddings[embeddingIndex]);

                var embeddingDetails = new EmbeddingDetails
                {
                    EntryName = entry.EntryName,
                    ShardName = shardName,
                    VectorOffset = _vectorRepository.Count(shardName) - 1
                };

                _embeddingDetailsRepository.Upsert(shardName, embeddingDetails);
                totalEmbeddingsStored++;
            }
        }

        _logger.LogInformation("Added {Count} embeddings to index '{IndexName}'",
            totalEmbeddingsStored, indexName);

        return totalEmbeddingsStored;
    }

    public int ResetEmbeddings(string indexName)
    {
        _logger.LogInformation("Resetting embeddings for index '{IndexName}'", indexName);

        var indexDetails = _indexDetailsRepository.Read(indexName);
        if (indexDetails is null)
            throw new ItemNotFoundException($"Index '{indexName}' not found.");

        var totalReset = 0;

        for (var shardIndex = 0; shardIndex < indexDetails.NumberOfShards; shardIndex++)
        {
            var shardName = BuildShardName(indexName, shardIndex);
            var entryNames = _embeddingDetailsRepository.ListEntryNames(shardName);
            totalReset += entryNames.Count;

            _vectorRepository.DropIndex(shardName);
            _vectorRepository.InitializeIndex(shardName, indexDetails.Dimension);
            _embeddingDetailsRepository.DropTable(shardName);
            _embeddingDetailsRepository.InitializeTable(shardName);

            _shardDetailsRepository.Upsert(new ShardDetails
            {
                ShardName = shardName,
                IndexName = indexName,
                VectorCount = 0
            });
        }

        _logger.LogInformation("Reset {Count} embeddings for index '{IndexName}'", totalReset, indexName);
        return totalReset;
    }

    private string BuildShardName(string indexName, int shardIndex)
    {
        var shardSuffix = shardIndex == 0
            ? _configuration.DefaultShardName
            : $"shard_{shardIndex}";
        return $"{indexName}:{shardSuffix}";
    }
}