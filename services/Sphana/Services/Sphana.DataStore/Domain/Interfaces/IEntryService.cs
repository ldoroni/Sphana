namespace Sphana.DataStore.Domain.Interfaces;

/// <summary>
/// Service interface for entry management operations.
/// </summary>
public interface IEntryService
{
    IReadOnlyList<string> ListEntries(string indexName);
    void DeleteEntry(string indexName, string entryName);
}