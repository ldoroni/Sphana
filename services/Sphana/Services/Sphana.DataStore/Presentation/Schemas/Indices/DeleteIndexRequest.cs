using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Indices;

public sealed class DeleteIndexRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }
}