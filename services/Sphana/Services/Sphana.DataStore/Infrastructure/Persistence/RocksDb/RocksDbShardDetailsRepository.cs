using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Infrastructure.Persistence.RocksDb;

/// <summary>
/// RocksDB-backed repository for ShardDetails metadata.
/// </summary>
public sealed class RocksDbShardDetailsRepository : IShardDetailsRepository
{
    private const string TableName = "shard_details";
    private readonly IDocumentRepository<ShardDetails> _documentRepository;

    public RocksDbShardDetailsRepository(IDocumentRepository<ShardDetails> documentRepository)
    {
        _documentRepository = documentRepository;
        _documentRepository.InitializeTable(TableName);
    }

    public void Upsert(ShardDetails shardDetails)
    {
        _documentRepository.Upsert(TableName, shardDetails.ShardName, shardDetails);
    }

    public ShardDetails? Read(string shardName)
    {
        return _documentRepository.Read(TableName, shardName);
    }

    public void Delete(string shardName)
    {
        _documentRepository.Delete(TableName, shardName);
    }
}