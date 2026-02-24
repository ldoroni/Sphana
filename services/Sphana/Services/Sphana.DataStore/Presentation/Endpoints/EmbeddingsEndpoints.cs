using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Presentation.Schemas.Embeddings;

namespace Sphana.DataStore.Presentation.Endpoints;

/// <summary>
/// Minimal API endpoints for embedding management operations.
/// </summary>
public static class EmbeddingsEndpoints
{
    public static void MapEmbeddingsEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("v1/embeddings");

        group.MapPost("add", (AddEmbeddingsRequest request, IEmbeddingService embeddingService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Adding embeddings to index '{IndexName}' with {Count} entries",
                request.IndexName, request.Entries.Count);

            var domainEntries = request.Entries.Select(entry => new EmbeddingEntry
            {
                EntryName = entry.EntryName,
                Embeddings = entry.Embeddings
            }).ToList();

            int entriesCount = embeddingService.AddEmbeddings(request.IndexName, domainEntries);

            return Results.Ok(new AddEmbeddingsResponse { EntriesCount = entriesCount });
        });

        group.MapPost("reset", (ResetEmbeddingsRequest request, IEmbeddingService embeddingService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Resetting embeddings for index '{IndexName}'", request.IndexName);

            int entriesCount = embeddingService.ResetEmbeddings(request.IndexName);

            return Results.Ok(new ResetEmbeddingsResponse { EntriesCount = entriesCount });
        });
    }
}