using Grpc.Core;
using Microsoft.Extensions.Logging;
using Moq;
using Sphana.Database.Controllers;
using Sphana.Database.RPC.V1;
using Sphana.Database.Services;

namespace Sphana.Database.Tests.UnitTests.Controllers;

public class SphanaDatabaseGrpcControllerErrorTests
{
    private readonly Mock<IDocumentIngestionService> _mockIngestionService;
    private readonly Mock<IQueryService> _mockQueryService;
    private readonly Mock<ILogger<SphanaDatabaseGrpcController>> _mockLogger;
    private readonly SphanaDatabaseGrpcController _service;

    public SphanaDatabaseGrpcControllerErrorTests()
    {
        _mockIngestionService = new Mock<IDocumentIngestionService>();
        _mockQueryService = new Mock<IQueryService>();
        _mockLogger = new Mock<ILogger<SphanaDatabaseGrpcController>>();

        _service = new SphanaDatabaseGrpcController(
            _mockIngestionService.Object,
            _mockQueryService.Object,
            _mockLogger.Object);
    }

    [Fact]
    public void Constructor_WithNullIngestionService_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new SphanaDatabaseGrpcController(
            null!,
            _mockQueryService.Object,
            _mockLogger.Object));
    }

    [Fact]
    public void Constructor_WithNullQueryService_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new SphanaDatabaseGrpcController(
            _mockIngestionService.Object,
            null!,
            _mockLogger.Object));
    }

    [Fact]
    public void Constructor_WithNullLogger_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new SphanaDatabaseGrpcController(
            _mockIngestionService.Object,
            _mockQueryService.Object,
            null!));
    }

    [Fact]
    public async Task Ingest_WithNullIndex_ShouldReturnInvalidRequestError()
    {
        // Arrange
        var request = new IngestRequest
        {
            Index = null,
            Document = new RPC.V1.Document
            {
                Title = "Test",
                Document_ = "Content"
            }
        };

        var context = new Mock<ServerCallContext>();

        // Act
        var response = await _service.Ingest(request, context.Object);

        // Assert
        response.Should().NotBeNull();
        response.Status.Succeed.Should().BeFalse();
        response.Status.StatusCode.Should().Be("INVALID_REQUEST");
    }

    [Fact]
    public async Task Ingest_WithEmptyTenantId_ShouldReturnInvalidRequestError()
    {
        // Arrange
        var request = new IngestRequest
        {
            Index = new RPC.V1.Index
            {
                TenantId = "",
                IndexName = "test_index"
            },
            Document = new RPC.V1.Document
            {
                Title = "Test",
                Document_ = "Content"
            }
        };

        var context = new Mock<ServerCallContext>();

        // Act
        var response = await _service.Ingest(request, context.Object);

        // Assert
        response.Should().NotBeNull();
        response.Status.Succeed.Should().BeFalse();
        response.Status.StatusCode.Should().Be("INVALID_REQUEST");
        response.Status.Message.Should().Contain("Tenant ID is required");
    }

    [Fact]
    public async Task Ingest_WithNullDocument_ShouldReturnInvalidRequestError()
    {
        // Arrange
        var request = new IngestRequest
        {
            Index = new RPC.V1.Index
            {
                TenantId = "test_tenant",
                IndexName = "test_index"
            },
            Document = null
        };

        var context = new Mock<ServerCallContext>();

        // Act
        var response = await _service.Ingest(request, context.Object);

        // Assert
        response.Should().NotBeNull();
        response.Status.Succeed.Should().BeFalse();
        response.Status.StatusCode.Should().Be("INVALID_REQUEST");
        response.Status.Message.Should().Contain("Document content is required");
    }

    [Fact]
    public async Task Ingest_WithEmptyDocumentContent_ShouldReturnInvalidRequestError()
    {
        // Arrange
        var request = new IngestRequest
        {
            Index = new RPC.V1.Index
            {
                TenantId = "test_tenant",
                IndexName = "test_index"
            },
            Document = new RPC.V1.Document
            {
                Title = "Test",
                Document_ = ""
            }
        };

        var context = new Mock<ServerCallContext>();

        // Act
        var response = await _service.Ingest(request, context.Object);

        // Assert
        response.Should().NotBeNull();
        response.Status.Succeed.Should().BeFalse();
        response.Status.StatusCode.Should().Be("INVALID_REQUEST");
        response.Status.Message.Should().Contain("Document content is required");
    }

    [Fact]
    public async Task Ingest_WhenIngestionServiceThrows_ShouldReturnInternalError()
    {
        // Arrange
        var request = new IngestRequest
        {
            Index = new RPC.V1.Index
            {
                TenantId = "test_tenant",
                IndexName = "test_index"
            },
            Document = new RPC.V1.Document
            {
                Title = "Test",
                Document_ = "Content"
            }
        };

        _mockIngestionService.Setup(x => x.IngestDocumentAsync(
                It.IsAny<Database.Models.Document>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new Exception("Ingestion failed"));

        var context = new Mock<ServerCallContext>();

        // Act
        var response = await _service.Ingest(request, context.Object);

        // Assert
        response.Should().NotBeNull();
        response.Status.Succeed.Should().BeFalse();
        response.Status.StatusCode.Should().Be("INTERNAL_ERROR");
        response.Status.Message.Should().Contain("Ingestion failed");
    }

    [Fact]
    public async Task Query_WithNullIndex_ShouldReturnInvalidRequestError()
    {
        // Arrange
        var request = new QueryRequest
        {
            Index = null,
            Query = "test query"
        };

        var context = new Mock<ServerCallContext>();

        // Act
        var response = await _service.Query(request, context.Object);

        // Assert
        response.Should().NotBeNull();
        response.Status.Succeed.Should().BeFalse();
        response.Status.StatusCode.Should().Be("INVALID_REQUEST");
    }

    [Fact]
    public async Task Query_WithEmptyTenantId_ShouldReturnInvalidRequestError()
    {
        // Arrange
        var request = new QueryRequest
        {
            Index = new RPC.V1.Index
            {
                TenantId = "",
                IndexName = "test_index"
            },
            Query = "test query"
        };

        var context = new Mock<ServerCallContext>();

        // Act
        var response = await _service.Query(request, context.Object);

        // Assert
        response.Should().NotBeNull();
        response.Status.Succeed.Should().BeFalse();
        response.Status.StatusCode.Should().Be("INVALID_REQUEST");
        response.Status.Message.Should().Contain("Tenant ID is required");
    }

    [Fact]
    public async Task Query_WithEmptyQuery_ShouldReturnInvalidRequestError()
    {
        // Arrange
        var request = new QueryRequest
        {
            Index = new RPC.V1.Index
            {
                TenantId = "test_tenant",
                IndexName = "test_index"
            },
            Query = ""
        };

        var context = new Mock<ServerCallContext>();

        // Act
        var response = await _service.Query(request, context.Object);

        // Assert
        response.Should().NotBeNull();
        response.Status.Succeed.Should().BeFalse();
        response.Status.StatusCode.Should().Be("INVALID_REQUEST");
        response.Status.Message.Should().Contain("Query is required");
    }

    [Fact]
    public async Task Query_WhenQueryServiceThrows_ShouldReturnInternalError()
    {
        // Arrange
        var request = new QueryRequest
        {
            Index = new RPC.V1.Index
            {
                TenantId = "test_tenant",
                IndexName = "test_index"
            },
            Query = "test query"
        };

        _mockQueryService.Setup(x => x.ExecuteQueryAsync(
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new Exception("Query failed"));

        var context = new Mock<ServerCallContext>();

        // Act
        var response = await _service.Query(request, context.Object);

        // Assert
        response.Should().NotBeNull();
        response.Status.Succeed.Should().BeFalse();
        response.Status.StatusCode.Should().Be("INTERNAL_ERROR");
        response.Status.Message.Should().Contain("Query failed");
    }
}

