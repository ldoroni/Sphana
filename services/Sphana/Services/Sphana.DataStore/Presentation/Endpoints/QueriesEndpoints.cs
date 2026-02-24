using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Presentation.Schemas.Queries;

namespace Sphana.DataStore.Presentation.Endpoints;

/// <summary>
/// Minimal API endpoints for query execution operations.
/// </summary>
public static class QueriesEndpoints
{
    public static void MapQueriesEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("v1/queries");

        group.MapPost("execute", (ExecuteQueryRequest request, IQueryService queryService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Executing query on index '{IndexName}' with limit {Limit}",
                request.IndexName, request.Limit);

            var results = queryService.ExecuteQuery(request.IndexName, request.Embeddings, request.Limit);

            var response = results.Select(result => new ExecuteQueryResultSchema
            {
                EntryName = result.EntryName,
                Distance = result.Distance,
                Payload = result.PayloadBase64
            }).ToList();

            return Results.Ok(response);
        });
    }
}