using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Entries;

public sealed class ListEntriesResponse
{
    [JsonPropertyName("entry_names")]
    public required IReadOnlyList<string> EntryNames { get; init; }

    [JsonPropertyName("entries_count")]
    public required int EntriesCount { get; init; }
}