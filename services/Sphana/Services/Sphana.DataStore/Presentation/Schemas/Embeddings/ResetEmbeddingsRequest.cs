using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Embeddings;

public sealed class ResetEmbeddingsRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }
}