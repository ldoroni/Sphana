using System.Net;

namespace Sphana.ManagedException;

/// <summary>
/// Base exception class for all managed (domain) exceptions in the platform.
/// Maps to HTTP status codes with structured error details.
/// </summary>
public abstract class ManagedException : Exception
{
    public ErrorDetails ErrorDetails { get; }

    protected ManagedException(ErrorDetails errorDetails)
        : base(errorDetails.Message)
    {
        ErrorDetails = errorDetails;
    }

    protected ManagedException(ErrorDetails errorDetails, Exception innerException)
        : base(errorDetails.Message, innerException)
    {
        ErrorDetails = errorDetails;
    }

    protected ManagedException(
        HttpStatusCode statusCode,
        string diagnosticCode,
        string message,
        Dictionary<string, string>? diagnosticDetails = null)
        : base(message)
    {
        ErrorDetails = new ErrorDetails(
            statusCode,
            diagnosticCode,
            diagnosticDetails ?? new Dictionary<string, string>(),
            message);
    }
}