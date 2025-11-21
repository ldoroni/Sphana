using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Logging;
using Moq;
using Sphana.Database.Infrastructure.GraphStorage;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Services;

namespace Sphana.Database.Tests.Services;

public class SphanaDatabaseHealthCheckTests
{
    private readonly Mock<IVectorIndex> _mockVectorIndex;
    private readonly Mock<IGraphStorage> _mockGraphStorage;
    private readonly Mock<ILogger<SphanaDatabaseHealthCheck>> _mockLogger;
    private readonly SphanaDatabaseHealthCheck _healthCheck;

    public SphanaDatabaseHealthCheckTests()
    {
        _mockVectorIndex = new Mock<IVectorIndex>();
        _mockGraphStorage = new Mock<IGraphStorage>();
        _mockLogger = new Mock<ILogger<SphanaDatabaseHealthCheck>>();
        
        _healthCheck = new SphanaDatabaseHealthCheck(
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object);
    }

    [Fact]
    public async Task CheckHealthAsync_WhenComponentsAreNotNull_ShouldReturnHealthy()
    {
        // Arrange
        var context = new HealthCheckContext();

        // Act
        var result = await _healthCheck.CheckHealthAsync(context, CancellationToken.None);

        // Assert
        result.Status.Should().Be(HealthStatus.Healthy);
        result.Description.Should().Contain("Sphana Database is running");
        result.Data.Should().ContainKey("vectorIndexLoaded");
        result.Data.Should().ContainKey("graphStorageLoaded");
        result.Data.Should().ContainKey("timestamp");
    }

    [Fact]
    public async Task CheckHealthAsync_ShouldIncludeTimestamp()
    {
        // Arrange
        var context = new HealthCheckContext();

        // Act
        var result = await _healthCheck.CheckHealthAsync(context, CancellationToken.None);

        // Assert
        result.Data["timestamp"].Should().BeOfType<DateTime>();
        ((DateTime)result.Data["timestamp"]).Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(2));
    }

    [Fact]
    public async Task CheckHealthAsync_WithCancellationToken_ShouldComplete()
    {
        // Arrange
        var context = new HealthCheckContext();
        var cts = new CancellationTokenSource();

        // Act
        var result = await _healthCheck.CheckHealthAsync(context, cts.Token);

        // Assert
        result.Status.Should().Be(HealthStatus.Healthy);
    }

    [Fact]
    public void Constructor_WithNullVectorIndex_ShouldNotThrow()
    {
        // Act & Assert - the constructor accepts nulls and the health check handles them
        Action act = () => new SphanaDatabaseHealthCheck(
            null!,
            _mockGraphStorage.Object,
            _mockLogger.Object);

        // The constructor doesn't validate nulls, so this should not throw
        act.Should().NotThrow();
    }
}

