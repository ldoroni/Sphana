using System.Net;

namespace Sphana.ManagedException.Authorization;

/// <summary>
/// Thrown when a request lacks valid authentication credentials.
/// </summary>
public class UnauthenticatedException : ManagedException
{
    public UnauthenticatedException(string message, Dictionary<string, string>? diagnosticDetails = null)
        : base(HttpStatusCode.Unauthorized, "UNAUTHENTICATED", message, diagnosticDetails)
    {
    }
}