using Microsoft.Extensions.Logging;
using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Domain.Models;
using Sphana.DataStore.Infrastructure.Configuration;
using Sphana.ManagedException.Arguments;

namespace Sphana.DataStore.Application.Services;

/// <summary>
/// Manages index lifecycle: create, get, list, update, delete.
/// </summary>
public sealed class IndexService : IIndexService
{
    private readonly IIndexDetailsRepository _indexDetailsRepository;
    private readonly IShardDetailsRepository _shardDetailsRepository;
    private readonly IEmbeddingDetailsRepository _embeddingDetailsRepository;
    private readonly IVectorRepository _vectorRepository;
    private readonly IBlobRepository _blobRepository;
    private readonly DataStoreConfiguration _configuration;
    private readonly ILogger<IndexService> _logger;

    public IndexService(
        IIndexDetailsRepository indexDetailsRepository,
        IShardDetailsRepository shardDetailsRepository,
        IEmbeddingDetailsRepository embeddingDetailsRepository,
        IVectorRepository vectorRepository,
        IBlobRepository blobRepository,
        DataStoreConfiguration configuration,
        ILogger<IndexService> logger)
    {
        _indexDetailsRepository = indexDetailsRepository;
        _shardDetailsRepository = shardDetailsRepository;
        _embeddingDetailsRepository = embeddingDetailsRepository;
        _vectorRepository = vectorRepository;
        _blobRepository = blobRepository;
        _configuration = configuration;
        _logger = logger;
    }

    public IndexDetails CreateIndex(string indexName, string description, string mediaType, int dimension, int numberOfShards)
    {
        _logger.LogInformation("Creating index '{IndexName}' with dimension {Dimension} and {NumberOfShards} shards",
            indexName, dimension, numberOfShards);

        if (_indexDetailsRepository.Exists(indexName))
            throw new ItemAlreadyExistsException($"Index '{indexName}' already exists.");

        var now = DateTime.UtcNow;
        var indexDetails = new IndexDetails
        {
            IndexName = indexName,
            Description = description,
            MediaType = mediaType,
            Dimension = dimension,
            NumberOfShards = numberOfShards,
            CreationTimestamp = now,
            ModificationTimestamp = now
        };

        _indexDetailsRepository.Upsert(indexDetails);

        for (var shardIndex = 0; shardIndex < numberOfShards; shardIndex++)
        {
            var shardName = BuildShardName(indexName, shardIndex);

            var shardDetails = new ShardDetails
            {
                ShardName = shardName,
                IndexName = indexName,
                VectorCount = 0
            };

            _shardDetailsRepository.Upsert(shardDetails);
            _embeddingDetailsRepository.InitializeTable(shardName);
            _vectorRepository.InitializeIndex(shardName, dimension);
        }

        _blobRepository.InitializeStorage(indexName);

        _logger.LogInformation("Index '{IndexName}' created successfully", indexName);
        return indexDetails;
    }

    public IndexDetails GetIndex(string indexName)
    {
        _logger.LogInformation("Retrieving index '{IndexName}'", indexName);

        var indexDetails = _indexDetailsRepository.Read(indexName);
        if (indexDetails is null)
            throw new ItemNotFoundException($"Index '{indexName}' not found.");

        return indexDetails;
    }

    public IReadOnlyList<IndexDetails> ListIndices()
    {
        _logger.LogInformation("Listing all indices");
        return _indexDetailsRepository.ListAll();
    }

    public IndexDetails UpdateIndex(string indexName, string description)
    {
        _logger.LogInformation("Updating index '{IndexName}'", indexName);

        var existingIndex = _indexDetailsRepository.Read(indexName);
        if (existingIndex is null)
            throw new ItemNotFoundException($"Index '{indexName}' not found.");

        var updatedIndex = new IndexDetails
        {
            IndexName = existingIndex.IndexName,
            Description = description,
            MediaType = existingIndex.MediaType,
            Dimension = existingIndex.Dimension,
            NumberOfShards = existingIndex.NumberOfShards,
            CreationTimestamp = existingIndex.CreationTimestamp,
            ModificationTimestamp = DateTime.UtcNow
        };

        _indexDetailsRepository.Upsert(updatedIndex);
        _logger.LogInformation("Index '{IndexName}' updated successfully", indexName);
        return updatedIndex;
    }

    public void DeleteIndex(string indexName)
    {
        _logger.LogInformation("Deleting index '{IndexName}'", indexName);

        var existingIndex = _indexDetailsRepository.Read(indexName);
        if (existingIndex is null)
            throw new ItemNotFoundException($"Index '{indexName}' not found.");

        for (var shardIndex = 0; shardIndex < existingIndex.NumberOfShards; shardIndex++)
        {
            var shardName = BuildShardName(indexName, shardIndex);
            _vectorRepository.DropIndex(shardName);
            _embeddingDetailsRepository.DropTable(shardName);
            _shardDetailsRepository.Delete(shardName);
        }

        _blobRepository.DropStorage(indexName);
        _indexDetailsRepository.Delete(indexName);

        _logger.LogInformation("Index '{IndexName}' deleted successfully", indexName);
    }

    public bool IndexExists(string indexName)
    {
        _logger.LogInformation("Checking existence of index '{IndexName}'", indexName);
        return _indexDetailsRepository.Exists(indexName);
    }

    private string BuildShardName(string indexName, int shardIndex)
    {
        var shardSuffix = shardIndex == 0
            ? _configuration.DefaultShardName
            : $"shard_{shardIndex}";
        return $"{indexName}:{shardSuffix}";
    }
}