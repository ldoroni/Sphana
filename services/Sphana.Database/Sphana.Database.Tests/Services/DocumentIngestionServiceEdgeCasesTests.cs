using Microsoft.Extensions.Logging;
using Moq;
using Sphana.Database.Infrastructure.GraphStorage;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Models;
using Sphana.Database.Services;

namespace Sphana.Database.Tests.Services;

public class DocumentIngestionServiceEdgeCasesTests
{
    private readonly Mock<IEmbeddingModel> _mockEmbeddingModel;
    private readonly Mock<IRelationExtractionModel> _mockRelationExtractionModel;
    private readonly Mock<IVectorIndex> _mockVectorIndex;
    private readonly Mock<IGraphStorage> _mockGraphStorage;
    private readonly Mock<ILogger<DocumentIngestionService>> _mockLogger;
    private readonly DocumentIngestionService _service;

    public DocumentIngestionServiceEdgeCasesTests()
    {
        _mockEmbeddingModel = new Mock<IEmbeddingModel>();
        _mockRelationExtractionModel = new Mock<IRelationExtractionModel>();
        _mockVectorIndex = new Mock<IVectorIndex>();
        _mockGraphStorage = new Mock<IGraphStorage>();
        _mockLogger = new Mock<ILogger<DocumentIngestionService>>();

        _service = new DocumentIngestionService(
            _mockEmbeddingModel.Object,
            _mockRelationExtractionModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            chunkSize: 512,
            chunkOverlap: 50,
            minRelationConfidence: 0.5f);
    }

    [Fact]
    public async Task IngestDocumentAsync_WithNullDocument_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(
            async () => await _service.IngestDocumentAsync(null!, CancellationToken.None));
    }

    [Fact]
    public async Task IngestDocumentAsync_WithEmptyContent_ShouldStillProcess()
    {
        // Arrange
        var document = new Document
        {
            Id = Guid.NewGuid().ToString(),
            TenantId = "test_tenant",
            IndexName = "test_index",
            Title = "Empty Doc",
            Content = "",
            ContentHash = "hash"
        };

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingsAsync(
                It.IsAny<string[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new[] { new float[128] });

        _mockVectorIndex.Setup(x => x.AddAsync(
                It.IsAny<string>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);

        // Act
        var result = await _service.IngestDocumentAsync(document, CancellationToken.None);

        // Assert
        result.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task IngestDocumentAsync_WithVeryLongContent_ShouldChunkCorrectly()
    {
        // Arrange
        var longContent = string.Join(" ", Enumerable.Repeat("word", 1000)); // ~5000 chars
        var document = new Document
        {
            Id = Guid.NewGuid().ToString(),
            TenantId = "test_tenant",
            IndexName = "test_index",
            Title = "Long Doc",
            Content = longContent,
            ContentHash = "hash"
        };

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingsAsync(
                It.IsAny<string[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync((string[] texts, CancellationToken _) => 
                texts.Select(_ => new float[128]).ToArray());

        _mockVectorIndex.Setup(x => x.AddAsync(
                It.IsAny<string>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);

        _mockRelationExtractionModel.Setup(x => x.ExtractRelationsAsync(
                It.IsAny<string>(),
                It.IsAny<List<ExtractedEntity>>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<ExtractedRelation>());

        // Act
        var result = await _service.IngestDocumentAsync(document, CancellationToken.None);

        // Assert
        result.Should().NotBeNullOrEmpty();
        _mockEmbeddingModel.Verify(x => x.GenerateEmbeddingsAsync(
            It.Is<string[]>(arr => arr.Length > 1), // Should have multiple chunks
            It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task IngestDocumentAsync_WhenEmbeddingFails_ShouldThrowException()
    {
        // Arrange
        var document = new Document
        {
            Id = Guid.NewGuid().ToString(),
            TenantId = "test_tenant",
            IndexName = "test_index",
            Title = "Test",
            Content = "Test content",
            ContentHash = "hash"
        };

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingsAsync(
                It.IsAny<string[]>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new Exception("Embedding failed"));

        // Act & Assert
        await Assert.ThrowsAsync<Exception>(
            async () => await _service.IngestDocumentAsync(document, CancellationToken.None));
    }

    [Fact]
    public async Task IngestDocumentAsync_WhenVectorIndexFails_ShouldThrowException()
    {
        // Arrange
        var document = new Document
        {
            Id = Guid.NewGuid().ToString(),
            TenantId = "test_tenant",
            IndexName = "test_index",
            Title = "Test",
            Content = "Test content",
            ContentHash = "hash"
        };

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingsAsync(
                It.IsAny<string[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new[] { new float[128] });

        _mockVectorIndex.Setup(x => x.AddAsync(
                It.IsAny<string>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new Exception("Index failed"));

        // Act & Assert
        await Assert.ThrowsAsync<Exception>(
            async () => await _service.IngestDocumentAsync(document, CancellationToken.None));
    }

    [Fact]
    public void Constructor_WithNullEmbeddingModel_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new DocumentIngestionService(
            null!,
            _mockRelationExtractionModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            512,
            50,
            0.5f));
    }

    [Fact]
    public void Constructor_WithNullRelationExtractionModel_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new DocumentIngestionService(
            _mockEmbeddingModel.Object,
            null!,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            512,
            50,
            0.5f));
    }

    [Fact]
    public void Constructor_WithNullVectorIndex_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new DocumentIngestionService(
            _mockEmbeddingModel.Object,
            _mockRelationExtractionModel.Object,
            null!,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            512,
            50,
            0.5f));
    }

    [Fact]
    public void Constructor_WithNullGraphStorage_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new DocumentIngestionService(
            _mockEmbeddingModel.Object,
            _mockRelationExtractionModel.Object,
            _mockVectorIndex.Object,
            null!,
            _mockLogger.Object,
            512,
            50,
            0.5f));
    }

    [Fact]
    public void Constructor_WithNullLogger_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new DocumentIngestionService(
            _mockEmbeddingModel.Object,
            _mockRelationExtractionModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            null!,
            512,
            50,
            0.5f));
    }
}

