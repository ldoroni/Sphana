using Sphana.DataStore.Domain.Models;

namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Repository for persisting and retrieving index metadata.
/// </summary>
public interface IIndexDetailsRepository
{
    void Upsert(IndexDetails indexDetails);
    IndexDetails? Read(string indexName);
    void Delete(string indexName);
    bool Exists(string indexName);
    IReadOnlyList<IndexDetails> ListAll();
}