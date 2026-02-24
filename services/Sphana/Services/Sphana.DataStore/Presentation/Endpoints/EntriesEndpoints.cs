using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Presentation.Schemas.Entries;

namespace Sphana.DataStore.Presentation.Endpoints;

/// <summary>
/// Minimal API endpoints for entry management operations.
/// </summary>
public static class EntriesEndpoints
{
    public static void MapEntriesEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("v1/entries");

        group.MapPost("list", (ListEntriesRequest request, IEntryService entryService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Listing entries for index '{IndexName}'", request.IndexName);

            IReadOnlyList<string> entryNames = entryService.ListEntries(request.IndexName);

            return Results.Ok(new ListEntriesResponse
            {
                EntryNames = entryNames,
                EntriesCount = entryNames.Count
            });
        });

        group.MapPost("delete", (DeleteEntryRequest request, IEntryService entryService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Deleting entry '{EntryName}' from index '{IndexName}'",
                request.EntryName, request.IndexName);

            entryService.DeleteEntry(request.IndexName, request.EntryName);

            return Results.Ok();
        });
    }
}