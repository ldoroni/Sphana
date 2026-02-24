using System.Net;

namespace Sphana.ManagedException.Internal;

/// <summary>
/// Thrown when an unexpected internal error occurs.
/// </summary>
public class InternalErrorException : ManagedException
{
    public InternalErrorException(string message, Dictionary<string, string>? diagnosticDetails = null)
        : base(HttpStatusCode.InternalServerError, "INTERNAL_ERROR", message, diagnosticDetails)
    {
    }
}