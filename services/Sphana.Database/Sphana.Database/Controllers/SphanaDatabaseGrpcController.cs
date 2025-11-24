using Grpc.Core;
using Sphana.Database.RPC.V1;
using Sphana.Database.Services;
using Sphana.Database.Models;
using Sphana.Common.RPC.V1;

namespace Sphana.Database.Controllers;

/// <summary>
/// gRPC service implementation for Sphana Database
/// </summary>
public sealed class SphanaDatabaseGrpcController : SphanaDatabase.SphanaDatabaseBase
{
    private readonly IDocumentIngestionService _ingestionService;
    private readonly IQueryService _queryService;
    private readonly ILogger<SphanaDatabaseGrpcController> _logger;

    public SphanaDatabaseGrpcController(
        IDocumentIngestionService ingestionService,
        IQueryService queryService,
        ILogger<SphanaDatabaseGrpcController> logger)
    {
        _ingestionService = ingestionService ?? throw new ArgumentNullException(nameof(ingestionService));
        _queryService = queryService ?? throw new ArgumentNullException(nameof(queryService));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public override async Task<IngestResponse> Ingest(
        IngestRequest request, 
        ServerCallContext context)
    {
        try
        {
            // Validate request before accessing properties
            if (request.Index == null || string.IsNullOrWhiteSpace(request.Index.TenantId))
            {
                return new IngestResponse
                {
                    Status = new Common.RPC.V1.Status
                    {
                        Succeed = false,
                        StatusCode = "INVALID_REQUEST",
                        Message = "Tenant ID is required"
                    }
                };
            }

            _logger.LogInformation("Received ingest request for index {IndexName} in tenant {TenantId}",
                request.Index.IndexName, request.Index.TenantId);

            if (request.Document == null || string.IsNullOrWhiteSpace(request.Document.Document_))
            {
                return new IngestResponse
                {
                    Status = new Common.RPC.V1.Status
                    {
                        Succeed = false,
                        StatusCode = "INVALID_REQUEST",
                        Message = "Document content is required"
                    }
                };
            }

            // Convert protobuf document to domain model
            var document = new Models.Document {
                Id = Guid.NewGuid().ToString(),
                TenantId = request.Index.TenantId,
                IndexName = request.Index.IndexName,
                Title = request.Document.Title,
                Content = request.Document.Document_,
                Metadata = request.Document.Metadata.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
                ContentHash = ComputeHash(request.Document.Document_)
            };

            // Ingest the document
            var documentId = await _ingestionService.IngestDocumentAsync(document, context.CancellationToken);

            _logger.LogInformation("Successfully ingested document {DocumentId}", documentId);

            return new IngestResponse
            {
                Status = new Common.RPC.V1.Status
                {
                    Succeed = true,
                    StatusCode = "OK",
                    Message = $"Document ingested successfully with ID: {documentId}"
                }
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing ingest request");

            return new IngestResponse
            {
                Status = new Common.RPC.V1.Status
                {
                    Succeed = false,
                    StatusCode = "INTERNAL_ERROR",
                    Message = $"Internal error: {ex.Message}"
                }
            };
        }
    }

    public override async Task<QueryResponse> Query(
        QueryRequest request, 
        ServerCallContext context)
    {
        try
        {
            // Validate request before accessing properties
            if (request.Index == null || string.IsNullOrWhiteSpace(request.Index.TenantId))
            {
                return new QueryResponse
                {
                    Status = new Common.RPC.V1.Status
                    {
                        Succeed = false,
                        StatusCode = "INVALID_REQUEST",
                        Message = "Tenant ID is required"
                    },
                    Result = string.Empty
                };
            }

            if (string.IsNullOrWhiteSpace(request.Query))
            {
                return new QueryResponse
                {
                    Status = new Common.RPC.V1.Status
                    {
                        Succeed = false,
                        StatusCode = "INVALID_REQUEST",
                        Message = "Query is required"
                    },
                    Result = string.Empty
                };
            }

            _logger.LogInformation("Received query request: {Query} for index {IndexName} in tenant {TenantId}",
                request.Query, request.Index.IndexName, request.Index.TenantId);

            // Execute the query
            var queryResult = await _queryService.ExecuteQueryAsync(
                request.Query,
                request.Index.TenantId,
                request.Index.IndexName,
                context.CancellationToken);

            _logger.LogInformation("Query completed in {Latency}ms with {ResultCount} results",
                queryResult.LatencyMs, queryResult.VectorResults.Count);

            return new QueryResponse
            {
                Status = new Common.RPC.V1.Status
                {
                    Succeed = true,
                    StatusCode = "OK",
                    Message = $"Query completed successfully in {queryResult.LatencyMs:F2}ms"
                },
                Result = queryResult.Answer
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing query request");

            return new QueryResponse
            {
                Status = new Common.RPC.V1.Status
                {
                    Succeed = false,
                    StatusCode = "INTERNAL_ERROR",
                    Message = $"Internal error: {ex.Message}"
                },
                Result = string.Empty
            };
        }
    }

    private static string ComputeHash(string content)
    {
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        var bytes = System.Text.Encoding.UTF8.GetBytes(content);
        var hash = sha256.ComputeHash(bytes);
        return Convert.ToBase64String(hash);
    }
}

