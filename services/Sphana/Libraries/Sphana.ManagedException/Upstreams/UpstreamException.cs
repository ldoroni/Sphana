using System.Net;

namespace Sphana.ManagedException.Upstreams;

/// <summary>
/// Thrown when an upstream service call fails.
/// </summary>
public class UpstreamException : ManagedException
{
    public UpstreamException(string message, Dictionary<string, string>? diagnosticDetails = null)
        : base(HttpStatusCode.BadGateway, "UPSTREAM_ERROR", message, diagnosticDetails)
    {
    }
}