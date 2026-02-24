using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Embeddings;

public sealed class AddEmbeddingsRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }

    [JsonPropertyName("entries")]
    public required IReadOnlyList<AddEmbeddingsEntrySchema> Entries { get; init; }
}

public sealed class AddEmbeddingsEntrySchema
{
    [JsonPropertyName("entry_name")]
    public required string EntryName { get; init; }

    [JsonPropertyName("embeddings")]
    public required IReadOnlyList<float[]> Embeddings { get; init; }
}