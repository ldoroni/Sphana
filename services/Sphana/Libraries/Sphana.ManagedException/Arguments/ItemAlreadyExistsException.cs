using System.Net;

namespace Sphana.ManagedException.Arguments;

/// <summary>
/// Thrown when attempting to create an item that already exists.
/// </summary>
public class ItemAlreadyExistsException : ManagedException
{
    public ItemAlreadyExistsException(string message, Dictionary<string, string>? diagnosticDetails = null)
        : base(HttpStatusCode.Conflict, "ITEM_ALREADY_EXISTS", message, diagnosticDetails)
    {
    }
}