using Microsoft.Extensions.Diagnostics.HealthChecks;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Infrastructure.GraphStorage;

namespace Sphana.Database.Services;

/// <summary>
/// Health check service for Sphana Database
/// </summary>
public class SphanaDatabaseHealthCheck : IHealthCheck
{
    private readonly IVectorIndex _vectorIndex;
    private readonly IGraphStorage _graphStorage;
    private readonly ILogger<SphanaDatabaseHealthCheck> _logger;

    public SphanaDatabaseHealthCheck(
        IVectorIndex vectorIndex,
        IGraphStorage graphStorage,
        ILogger<SphanaDatabaseHealthCheck> logger)
    {
        _vectorIndex = vectorIndex;
        _graphStorage = graphStorage;
        _logger = logger;
    }

    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var data = new Dictionary<string, object>
            {
                { "vectorIndexLoaded", _vectorIndex != null },
                { "graphStorageLoaded", _graphStorage != null },
                { "timestamp", DateTime.UtcNow }
            };

            return Task.FromResult(
                HealthCheckResult.Healthy("Sphana Database is running", data));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Health check failed");
            return Task.FromResult(
                HealthCheckResult.Unhealthy("Sphana Database health check failed", ex));
        }
    }
}

