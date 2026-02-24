using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Queries;

public sealed class ExecuteQueryResultSchema
{
    [JsonPropertyName("entry_name")]
    public required string EntryName { get; init; }

    [JsonPropertyName("distance")]
    public required float Distance { get; init; }

    [JsonPropertyName("payload")]
    public required string Payload { get; init; }
}