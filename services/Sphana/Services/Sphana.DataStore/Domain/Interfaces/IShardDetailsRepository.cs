using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Repository for persisting and retrieving shard metadata.
/// </summary>
public interface IShardDetailsRepository
{
    void Upsert(ShardDetails shardDetails);
    ShardDetails? Read(string shardName);
    void Delete(string shardName);
}