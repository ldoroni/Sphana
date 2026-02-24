using System.Net;

namespace Sphana.ManagedException;

/// <summary>
/// Contains structured error information for managed exceptions.
/// </summary>
public sealed class ErrorDetails
{
    public HttpStatusCode StatusCode { get; }
    public string DiagnosticCode { get; }
    public Dictionary<string, string> DiagnosticDetails { get; }
    public string Message { get; }

    public ErrorDetails(
        HttpStatusCode statusCode,
        string diagnosticCode,
        Dictionary<string, string> diagnosticDetails,
        string message)
    {
        StatusCode = statusCode;
        DiagnosticCode = diagnosticCode;
        DiagnosticDetails = diagnosticDetails;
        Message = message;
    }
}