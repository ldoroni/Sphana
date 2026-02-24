using System.Text.Json.Serialization;

namespace Sphana.RequestHandler;

/// <summary>
/// Standardized error response returned by the API when a managed exception occurs.
/// </summary>
public sealed class ErrorResponse
{
    [JsonPropertyName("diagnostic_code")]
    public string DiagnosticCode { get; init; } = string.Empty;

    [JsonPropertyName("message")]
    public string Message { get; init; } = string.Empty;

    [JsonPropertyName("diagnostic_details")]
    public Dictionary<string, string> DiagnosticDetails { get; init; } = new();
}