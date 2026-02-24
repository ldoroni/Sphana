using System.Net;

namespace Sphana.ManagedException.Arguments;

/// <summary>
/// Thrown when an invalid argument is provided to an operation.
/// </summary>
public class InvalidArgumentException : ManagedException
{
    public InvalidArgumentException(string message, Dictionary<string, string>? diagnosticDetails = null)
        : base(HttpStatusCode.BadRequest, "INVALID_ARGUMENT", message, diagnosticDetails)
    {
    }
}