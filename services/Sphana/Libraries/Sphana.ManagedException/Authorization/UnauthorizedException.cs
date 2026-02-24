using System.Net;

namespace Sphana.ManagedException.Authorization;

/// <summary>
/// Thrown when a request is authenticated but lacks sufficient permissions.
/// </summary>
public class UnauthorizedException : ManagedException
{
    public UnauthorizedException(string message, Dictionary<string, string>? diagnosticDetails = null)
        : base(HttpStatusCode.Forbidden, "UNAUTHORIZED", message, diagnosticDetails)
    {
    }
}