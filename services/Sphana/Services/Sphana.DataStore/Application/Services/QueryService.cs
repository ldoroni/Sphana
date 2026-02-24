using Microsoft.Extensions.Logging;
using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Domain.Models;
using Sphana.DataStore.Infrastructure.Configuration;
using Sphana.ManagedException.Arguments;

namespace Sphana.DataStore.Application.Services;

/// <summary>
/// Executes vector similarity queries across index shards and enriches results with payloads.
/// </summary>
public sealed class QueryService : IQueryService
{
    private readonly IIndexDetailsRepository _indexDetailsRepository;
    private readonly IVectorRepository _vectorRepository;
    private readonly IBlobRepository _blobRepository;
    private readonly DataStoreConfiguration _configuration;
    private readonly ILogger<QueryService> _logger;

    public QueryService(
        IIndexDetailsRepository indexDetailsRepository,
        IVectorRepository vectorRepository,
        IBlobRepository blobRepository,
        DataStoreConfiguration configuration,
        ILogger<QueryService> logger)
    {
        _indexDetailsRepository = indexDetailsRepository;
        _vectorRepository = vectorRepository;
        _blobRepository = blobRepository;
        _configuration = configuration;
        _logger = logger;
    }

    public IReadOnlyList<ExecuteQueryResult> ExecuteQuery(string indexName, float[] embeddings, int limit)
    {
        _logger.LogInformation("Executing query on index '{IndexName}' with limit {Limit}", indexName, limit);

        var indexDetails = _indexDetailsRepository.Read(indexName);
        if (indexDetails is null)
            throw new ItemNotFoundException($"Index '{indexName}' not found.");

        var allResults = new List<EmbeddingResult>();

        for (var shardIndex = 0; shardIndex < indexDetails.NumberOfShards; shardIndex++)
        {
            var shardName = BuildShardName(indexName, shardIndex);
            var shardResults = _vectorRepository.Search(shardName, embeddings, limit);
            allResults.AddRange(shardResults);
        }

        var topResults = allResults
            .OrderBy(result => result.Distance)
            .Take(limit)
            .ToList();

        var queryResults = new List<ExecuteQueryResult>(topResults.Count);

        foreach (var embeddingResult in topResults)
        {
            var entryName = embeddingResult.EntryName.Contains(':')
                ? embeddingResult.EntryName[..embeddingResult.EntryName.LastIndexOf(':')]
                : embeddingResult.EntryName;

            var payloadBytes = _blobRepository.ReadBlob(indexName, entryName);
            var payloadBase64 = payloadBytes is not null
                ? Convert.ToBase64String(payloadBytes)
                : string.Empty;

            queryResults.Add(new ExecuteQueryResult
            {
                EntryName = entryName,
                Distance = embeddingResult.Distance,
                PayloadBase64 = payloadBase64
            });
        }

        _logger.LogInformation("Query on index '{IndexName}' returned {Count} results", indexName, queryResults.Count);
        return queryResults;
    }

    private string BuildShardName(string indexName, int shardIndex)
    {
        var shardSuffix = shardIndex == 0
            ? _configuration.DefaultShardName
            : $"shard_{shardIndex}";
        return $"{indexName}:{shardSuffix}";
    }
}