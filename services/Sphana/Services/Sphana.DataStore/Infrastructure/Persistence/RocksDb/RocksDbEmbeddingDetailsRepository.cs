using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Infrastructure.Persistence.RocksDb;

/// <summary>
/// RocksDB-backed repository for EmbeddingDetails metadata, scoped per index table.
/// </summary>
public sealed class RocksDbEmbeddingDetailsRepository : IEmbeddingDetailsRepository
{
    private readonly IDocumentRepository<EmbeddingDetails> _documentRepository;

    public RocksDbEmbeddingDetailsRepository(IDocumentRepository<EmbeddingDetails> documentRepository)
    {
        _documentRepository = documentRepository;
    }

    public void InitializeTable(string tableName)
    {
        _documentRepository.InitializeTable(tableName);
    }

    public void DropTable(string tableName)
    {
        _documentRepository.DropTable(tableName);
    }

    public void Upsert(string shardName, EmbeddingDetails embeddingDetails)
    {
        _documentRepository.Upsert(shardName, embeddingDetails.EntryName, embeddingDetails);
    }

    public EmbeddingDetails? Read(string shardName, string entryName)
    {
        return _documentRepository.Read(shardName, entryName);
    }

    public void Delete(string shardName, string entryName)
    {
        _documentRepository.Delete(shardName, entryName);
    }

    public IReadOnlyList<string> ListEntryNames(string shardName)
    {
        return _documentRepository.ListKeys(shardName);
    }
}