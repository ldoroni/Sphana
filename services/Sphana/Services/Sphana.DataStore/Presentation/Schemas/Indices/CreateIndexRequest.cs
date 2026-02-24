using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Indices;

public sealed class CreateIndexRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }

    [JsonPropertyName("description")]
    public required string Description { get; init; }

    [JsonPropertyName("media_type")]
    public required string MediaType { get; init; }

    [JsonPropertyName("dimension")]
    public required int Dimension { get; init; }

    [JsonPropertyName("number_of_shards")]
    public required int NumberOfShards { get; init; }
}