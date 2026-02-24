using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Domain.Models;
using Sphana.DataStore.Presentation.Schemas.Indices;

namespace Sphana.DataStore.Presentation.Endpoints;

/// <summary>
/// Minimal API endpoints for index management operations.
/// </summary>
public static class IndicesEndpoints
{
    public static void MapIndicesEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("v1/indices");

        group.MapPost("create", (CreateIndexRequest request, IIndexService indexService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Creating index '{IndexName}'", request.IndexName);

            IndexDetails indexDetails = indexService.CreateIndex(
                request.IndexName,
                request.Description,
                request.MediaType,
                request.Dimension,
                request.NumberOfShards);

            return Results.Ok(MapToResponse(indexDetails));
        });

        group.MapPost("get", (GetIndexRequest request, IIndexService indexService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Getting index '{IndexName}'", request.IndexName);

            IndexDetails indexDetails = indexService.GetIndex(request.IndexName);

            return Results.Ok(MapToResponse(indexDetails));
        });

        group.MapPost("list", (IIndexService indexService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Listing all indices");

            IReadOnlyList<IndexDetails> allIndices = indexService.ListIndices();

            List<IndexDetailsResponse> responseList = allIndices
                .Select(MapToResponse)
                .ToList();

            return Results.Ok(responseList);
        });

        group.MapPost("update", (UpdateIndexRequest request, IIndexService indexService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Updating index '{IndexName}'", request.IndexName);

            IndexDetails indexDetails = indexService.UpdateIndex(request.IndexName, request.Description);

            return Results.Ok(MapToResponse(indexDetails));
        });

        group.MapPost("delete", (DeleteIndexRequest request, IIndexService indexService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Deleting index '{IndexName}'", request.IndexName);

            indexService.DeleteIndex(request.IndexName);

            return Results.Ok();
        });

        group.MapPost("exists", (IndexExistsRequest request, IIndexService indexService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Checking existence of index '{IndexName}'", request.IndexName);

            bool exists = indexService.IndexExists(request.IndexName);

            return Results.Ok(new IndexExistsResponse { Exists = exists });
        });
    }

    private static IndexDetailsResponse MapToResponse(IndexDetails indexDetails)
    {
        return new IndexDetailsResponse
        {
            IndexName = indexDetails.IndexName,
            Description = indexDetails.Description,
            MediaType = indexDetails.MediaType,
            Dimension = indexDetails.Dimension,
            NumberOfShards = indexDetails.NumberOfShards,
            CreationTimestamp = indexDetails.CreationTimestamp,
            ModificationTimestamp = indexDetails.ModificationTimestamp
        };
    }
}