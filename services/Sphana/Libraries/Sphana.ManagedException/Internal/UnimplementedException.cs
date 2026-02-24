using System.Net;

namespace Sphana.ManagedException.Internal;

/// <summary>
/// Thrown when a requested operation is not yet implemented.
/// </summary>
public class UnimplementedException : ManagedException
{
    public UnimplementedException(string message, Dictionary<string, string>? diagnosticDetails = null)
        : base(HttpStatusCode.NotImplemented, "UNIMPLEMENTED", message, diagnosticDetails)
    {
    }
}