using Microsoft.Extensions.Logging;
using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Infrastructure.Persistence.RocksDb;

/// <summary>
/// RocksDB-backed repository for IndexDetails metadata.
/// </summary>
public sealed class RocksDbIndexDetailsRepository : IIndexDetailsRepository
{
    private const string TableName = "index_details";
    private readonly IDocumentRepository<IndexDetails> _documentRepository;

    public RocksDbIndexDetailsRepository(IDocumentRepository<IndexDetails> documentRepository)
    {
        _documentRepository = documentRepository;
        _documentRepository.InitializeTable(TableName);
    }

    public void Upsert(IndexDetails indexDetails)
    {
        _documentRepository.Upsert(TableName, indexDetails.IndexName, indexDetails);
    }

    public IndexDetails? Read(string indexName)
    {
        return _documentRepository.Read(TableName, indexName);
    }

    public void Delete(string indexName)
    {
        _documentRepository.Delete(TableName, indexName);
    }

    public bool Exists(string indexName)
    {
        return _documentRepository.Exists(TableName, indexName);
    }

    public IReadOnlyList<IndexDetails> ListAll()
    {
        var keys = _documentRepository.ListKeys(TableName);
        var results = new List<IndexDetails>();
        foreach (var key in keys)
        {
            var item = _documentRepository.Read(TableName, key);
            if (item is not null)
                results.Add(item);
        }
        return results;
    }
}