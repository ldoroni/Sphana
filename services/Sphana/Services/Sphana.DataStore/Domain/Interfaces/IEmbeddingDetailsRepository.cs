using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Repository for persisting and retrieving embedding metadata within shards.
/// </summary>
public interface IEmbeddingDetailsRepository
{
    void InitializeTable(string shardName);
    void DropTable(string shardName);
    void Upsert(string shardName, EmbeddingDetails embeddingDetails);
    EmbeddingDetails? Read(string shardName, string entryName);
    void Delete(string shardName, string entryName);
    IReadOnlyList<string> ListEntryNames(string shardName);
}