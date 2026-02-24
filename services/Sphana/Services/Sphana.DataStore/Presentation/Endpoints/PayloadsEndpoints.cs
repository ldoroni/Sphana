using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Presentation.Schemas.Payloads;

namespace Sphana.DataStore.Presentation.Endpoints;

/// <summary>
/// Minimal API endpoints for payload management operations.
/// </summary>
public static class PayloadsEndpoints
{
    public static void MapPayloadsEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("v1/payloads");

        group.MapPost("upload", (UploadPayloadRequest request, IPayloadService payloadService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Uploading payload for entry '{EntryName}' in index '{IndexName}'",
                request.EntryName, request.IndexName);

            var payloadBytes = Convert.FromBase64String(request.PayloadBase64);
            payloadService.UploadPayload(request.IndexName, request.EntryName, payloadBytes);

            return Results.Ok(new PayloadResponse { EntryName = request.EntryName });
        });

        group.MapPost("append", (AppendPayloadRequest request, IPayloadService payloadService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Appending payload for entry '{EntryName}' in index '{IndexName}'",
                request.EntryName, request.IndexName);

            var payloadBytes = Convert.FromBase64String(request.PayloadBase64);
            payloadService.AppendPayload(request.IndexName, request.EntryName, payloadBytes);

            return Results.Ok(new PayloadResponse { EntryName = request.EntryName });
        });

        group.MapPost("delete", (DeletePayloadRequest request, IPayloadService payloadService, ILogger<Program> logger) =>
        {
            logger.LogInformation("Deleting payload for entry '{EntryName}' in index '{IndexName}'",
                request.EntryName, request.IndexName);

            payloadService.DeletePayload(request.IndexName, request.EntryName);

            return Results.Ok(new PayloadResponse { EntryName = request.EntryName });
        });
    }
}