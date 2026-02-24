using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Queries;

public sealed class ExecuteQueryRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }

    [JsonPropertyName("embeddings")]
    public required float[] Embeddings { get; init; }

    [JsonPropertyName("limit")]
    public required int Limit { get; init; }
}