using Microsoft.Extensions.Logging;
using Moq;
using Sphana.Database.Infrastructure.GraphStorage;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Models.KnowledgeGraph;
using Sphana.Database.Services;

namespace Sphana.Database.Tests.Services;

public class QueryServiceEdgeCasesTests
{
    private readonly Mock<IEmbeddingModel> _mockEmbeddingModel;
    private readonly Mock<IGnnRankerModel> _mockGnnRankerModel;
    private readonly Mock<ILlmGeneratorModel> _mockLlmGeneratorModel;
    private readonly Mock<INerModel> _mockNerModel;
    private readonly Mock<IVectorIndex> _mockVectorIndex;
    private readonly Mock<IGraphStorage> _mockGraphStorage;
    private readonly Mock<ILogger<QueryService>> _mockLogger;
    private readonly QueryService _service;

    public QueryServiceEdgeCasesTests()
    {
        _mockEmbeddingModel = new Mock<IEmbeddingModel>();
        _mockGnnRankerModel = new Mock<IGnnRankerModel>();
        _mockLlmGeneratorModel = new Mock<ILlmGeneratorModel>();
        _mockNerModel = new Mock<INerModel>();
        _mockVectorIndex = new Mock<IVectorIndex>();
        _mockGraphStorage = new Mock<IGraphStorage>();
        _mockLogger = new Mock<ILogger<QueryService>>();

        // Setup defaults
        _mockNerModel.Setup(x => x.ExtractEntitiesAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<ExtractedEntity>());
        _mockLlmGeneratorModel.Setup(x => x.GenerateAnswerAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync("Generated Answer");

        _service = new QueryService(
            _mockEmbeddingModel.Object,
            _mockGnnRankerModel.Object,
            _mockLlmGeneratorModel.Object,
            _mockNerModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            vectorSearchWeight: 0.6f,
            graphSearchWeight: 0.4f,
            vectorSearchTopK: 20,
            maxSubgraphs: 10,
            maxGenerationTokens: 512);
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithNullQuery_ShouldThrowArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            async () => await _service.ExecuteQueryAsync(
                null!,
                "tenant",
                "index",
                CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithEmptyQuery_ShouldThrowArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            async () => await _service.ExecuteQueryAsync(
                "",
                "tenant",
                "index",
                CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithWhitespaceQuery_ShouldThrowArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            async () => await _service.ExecuteQueryAsync(
                "   ",
                "tenant",
                "index",
                CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithNullTenantId_ShouldThrowArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            async () => await _service.ExecuteQueryAsync(
                "query",
                null!,
                "index",
                CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithNullIndexName_ShouldThrowArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            async () => await _service.ExecuteQueryAsync(
                "query",
                "tenant",
                null!,
                CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteQueryAsync_WhenNoVectorResults_ShouldReturnEmptyAnswer()
    {
        // Arrange
        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new float[128]);

        _mockVectorIndex.Setup(x => x.SearchAsync(
                It.IsAny<float[]>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<SearchResult>()); // No results

        _mockGnnRankerModel.Setup(x => x.RankSubgraphsAsync(
                It.IsAny<List<KnowledgeSubgraph>>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<KnowledgeSubgraph>());

        // Act
        var result = await _service.ExecuteQueryAsync(
            "test query",
            "tenant",
            "index",
            CancellationToken.None);

        // Assert
        result.Should().NotBeNull();
        result.VectorResults.Should().BeEmpty();
    }

    [Fact]
    public async Task ExecuteQueryAsync_WhenEmbeddingFails_ShouldThrowException()
    {
        // Arrange
        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new Exception("Embedding failed"));

        // Act & Assert
        await Assert.ThrowsAsync<Exception>(
            async () => await _service.ExecuteQueryAsync(
                "test query",
                "tenant",
                "index",
                CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteQueryAsync_WhenVectorSearchFails_ShouldThrowException()
    {
        // Arrange
        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new float[128]);

        _mockVectorIndex.Setup(x => x.SearchAsync(
                It.IsAny<float[]>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new Exception("Search failed"));

        // Act & Assert
        await Assert.ThrowsAsync<Exception>(
            async () => await _service.ExecuteQueryAsync(
                "test query",
                "tenant",
                "index",
                CancellationToken.None));
    }

    [Fact]
    public void Constructor_WithNullEmbeddingModel_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryService(
            null!,
            _mockGnnRankerModel.Object,
            _mockLlmGeneratorModel.Object,
            _mockNerModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            0.6f,
            0.4f,
            20,
            10,
            512));
    }

    [Fact]
    public void Constructor_WithNullGnnRankerModel_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryService(
            _mockEmbeddingModel.Object,
            null!,
            _mockLlmGeneratorModel.Object,
            _mockNerModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            0.6f,
            0.4f,
            20,
            10,
            512));
    }

    [Fact]
    public void Constructor_WithNullLlmGeneratorModel_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryService(
            _mockEmbeddingModel.Object,
            _mockGnnRankerModel.Object,
            null!,
            _mockNerModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            0.6f,
            0.4f,
            20,
            10,
            512));
    }

    [Fact]
    public void Constructor_WithNullNerModel_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryService(
            _mockEmbeddingModel.Object,
            _mockGnnRankerModel.Object,
            _mockLlmGeneratorModel.Object,
            null!,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            0.6f,
            0.4f,
            20,
            10,
            512));
    }

    [Fact]
    public void Constructor_WithNullVectorIndex_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryService(
            _mockEmbeddingModel.Object,
            _mockGnnRankerModel.Object,
            _mockLlmGeneratorModel.Object,
            _mockNerModel.Object,
            null!,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            0.6f,
            0.4f,
            20,
            10,
            512));
    }

    [Fact]
    public void Constructor_WithNullGraphStorage_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryService(
            _mockEmbeddingModel.Object,
            _mockGnnRankerModel.Object,
            _mockLlmGeneratorModel.Object,
            _mockNerModel.Object,
            _mockVectorIndex.Object,
            null!,
            _mockLogger.Object,
            0.6f,
            0.4f,
            20,
            10,
            512));
    }

    [Fact]
    public void Constructor_WithNullLogger_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryService(
            _mockEmbeddingModel.Object,
            _mockGnnRankerModel.Object,
            _mockLlmGeneratorModel.Object,
            _mockNerModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            null!,
            0.6f,
            0.4f,
            20,
            10,
            512));
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithCancellationToken_ShouldRespectCancellation()
    {
        // Arrange
        var cts = new CancellationTokenSource();
        cts.Cancel();

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new OperationCanceledException());

        // Act & Assert
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await _service.ExecuteQueryAsync(
                "test query",
                "tenant",
                "index",
                cts.Token));
    }
}

