using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Indices;

public sealed class IndexExistsResponse
{
    [JsonPropertyName("exists")]
    public required bool Exists { get; init; }
}