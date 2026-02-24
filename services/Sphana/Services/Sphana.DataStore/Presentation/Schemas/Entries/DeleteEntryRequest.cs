using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Entries;

public sealed class DeleteEntryRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }

    [JsonPropertyName("entry_name")]
    public required string EntryName { get; init; }
}