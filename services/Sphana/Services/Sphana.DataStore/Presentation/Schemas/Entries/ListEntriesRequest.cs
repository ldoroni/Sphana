using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Entries;

public sealed class ListEntriesRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }
}