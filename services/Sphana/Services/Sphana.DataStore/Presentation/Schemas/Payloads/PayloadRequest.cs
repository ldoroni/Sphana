using System.Text.Json.Serialization;

namespace Sphana.DataStore.Presentation.Schemas.Payloads;

public sealed class UploadPayloadRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }

    [JsonPropertyName("entry_name")]
    public required string EntryName { get; init; }

    [JsonPropertyName("payload_base64")]
    public required string PayloadBase64 { get; init; }
}

public sealed class AppendPayloadRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }

    [JsonPropertyName("entry_name")]
    public required string EntryName { get; init; }

    [JsonPropertyName("payload_base64")]
    public required string PayloadBase64 { get; init; }
}

public sealed class DeletePayloadRequest
{
    [JsonPropertyName("index_name")]
    public required string IndexName { get; init; }

    [JsonPropertyName("entry_name")]
    public required string EntryName { get; init; }
}

public sealed class PayloadResponse
{
    [JsonPropertyName("entry_name")]
    public required string EntryName { get; init; }
}