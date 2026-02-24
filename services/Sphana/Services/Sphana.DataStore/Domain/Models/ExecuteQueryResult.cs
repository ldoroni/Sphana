namespace Sphana.DataStore.Domain.Models;

/// <summary>
/// Represents a single query result containing the entry identifier, distance score, and payload.
/// </summary>
public sealed class ExecuteQueryResult
{
    public required string EntryName { get; init; }
    public required float Distance { get; init; }
    public required string PayloadBase64 { get; init; }
}