using System.Net;

namespace Sphana.ManagedException.Arguments;

/// <summary>
/// Thrown when a requested item is not found.
/// </summary>
public class ItemNotFoundException : ManagedException
{
    public ItemNotFoundException(string message, Dictionary<string, string>? diagnosticDetails = null)
        : base(HttpStatusCode.NotFound, "ITEM_NOT_FOUND", message, diagnosticDetails)
    {
    }
}