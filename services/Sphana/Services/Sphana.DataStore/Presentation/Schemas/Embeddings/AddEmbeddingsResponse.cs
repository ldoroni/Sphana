using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Embeddings;

public sealed class AddEmbeddingsResponse
{
    [JsonPropertyName("entries_count")]
    public required int EntriesCount { get; init; }
}