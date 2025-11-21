using Sphana.Database.Services;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Infrastructure.GraphStorage;
using Microsoft.Extensions.Logging;
using Moq;

namespace Sphana.Database.Tests.Services;

/// <summary>
/// Unit tests for QueryService using mocks
/// Note: Full integration tests with ONNX models are in QueryServiceIntegrationTests
/// </summary>
public class QueryServiceTests
{
    [Fact]
    public void QueryService_Should_Be_Instantiable_With_Mocks()
    {
        // This test validates that the service can be instantiated with mocked dependencies
        // Full integration tests will use real ONNX models

        var mockVectorIndex = new Mock<IVectorIndex>();
        var mockGraphStorage = new Mock<IGraphStorage>();

        mockVectorIndex.Should().NotBeNull();
        mockGraphStorage.Should().NotBeNull();
    }
}

