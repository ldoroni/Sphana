using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Indices;

public sealed class IndexExistsRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }
}