using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Indices;

public sealed class UpdateIndexRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }

    [JsonPropertyName("description")]
    public required string Description { get; init; }
}