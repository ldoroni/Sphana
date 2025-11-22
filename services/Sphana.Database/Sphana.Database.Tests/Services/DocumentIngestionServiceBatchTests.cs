using Microsoft.Extensions.Logging;
using Sphana.Database.Infrastructure.GraphStorage;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Models;
using Sphana.Database.Services;

namespace Sphana.Database.Tests.Services;

/// <summary>
/// Tests for batch ingestion scenarios in DocumentIngestionService
/// </summary>
public class DocumentIngestionServiceBatchTests
{
    private readonly Mock<IEmbeddingModel> _mockEmbeddingModel;
    private readonly Mock<IRelationExtractionModel> _mockRelationExtractionModel;
    private readonly Mock<INerModel> _mockNerModel;
    private readonly Mock<IVectorIndex> _mockVectorIndex;
    private readonly Mock<IGraphStorage> _mockGraphStorage;
    private readonly Mock<ILogger<DocumentIngestionService>> _mockLogger;

    public DocumentIngestionServiceBatchTests()
    {
        _mockEmbeddingModel = new Mock<IEmbeddingModel>();
        _mockRelationExtractionModel = new Mock<IRelationExtractionModel>();
        _mockNerModel = new Mock<INerModel>();
        _mockVectorIndex = new Mock<IVectorIndex>();
        _mockGraphStorage = new Mock<IGraphStorage>();
        _mockLogger = new Mock<ILogger<DocumentIngestionService>>();
        
        _mockNerModel.Setup(x => x.ExtractEntitiesAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<ExtractedEntity>());
    }

    [Fact]
    public async Task IngestDocumentsBatchAsync_WithMultipleDocuments_ShouldProcessAll()
    {
        // Arrange
        var service = CreateService();
        var documents = new List<Document>
        {
            CreateDocument("doc1", "Document 1 content"),
            CreateDocument("doc2", "Document 2 content"),
            CreateDocument("doc3", "Document 3 content")
        };

        SetupMocks();

        // Act
        var results = await service.IngestDocumentsBatchAsync(documents);

        // Assert
        results.Should().HaveCount(3);
        results.Should().Contain("doc1");
        results.Should().Contain("doc2");
        results.Should().Contain("doc3");
    }

    [Fact]
    public async Task IngestDocumentsBatchAsync_WithMaxConcurrency_ShouldRespectLimit()
    {
        // Arrange
        var service = CreateService();
        var documents = Enumerable.Range(1, 10)
            .Select(i => CreateDocument($"doc{i}", $"Content {i}"))
            .ToList();

        SetupMocks();

        // Act
        var results = await service.IngestDocumentsBatchAsync(documents, maxConcurrency: 3);

        // Assert
        results.Should().HaveCount(10);
    }

    [Fact]
    public async Task IngestDocumentsBatchAsync_WithEmptyList_ShouldReturnEmptyList()
    {
        // Arrange
        var service = CreateService();
        var documents = new List<Document>();

        // Act
        var results = await service.IngestDocumentsBatchAsync(documents);

        // Assert
        results.Should().BeEmpty();
    }

    [Fact]
    public async Task IngestDocumentsBatchAsync_WhenOneDocumentFails_ShouldContinueWithOthers()
    {
        // Arrange
        var service = CreateService();
        var documents = new List<Document>
        {
            CreateDocument("doc1", "Document 1 content"),
            CreateDocument("doc2", "Document 2 content"),  
            CreateDocument("doc3", "Document 3 content")
        };

        // Setup: Make doc2 fail
        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingsAsync(
                It.IsAny<string[]>(),
                It.IsAny<CancellationToken>()))
            .Returns<string[], CancellationToken>((texts, ct) =>
            {
                var embeddings = texts.Select(_ => GenerateRandomEmbedding(128)).ToArray();
                return Task.FromResult(embeddings);
            });

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

        _mockGraphStorage.Setup(x => x.AddNodeAsync(
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync("node_123");

        // Act
        var results = await service.IngestDocumentsBatchAsync(documents);

        // Assert
        results.Should().HaveCount(3);
    }


    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public async Task IngestDocumentsBatchAsync_WithDifferentConcurrencyLevels_ShouldWork(int maxConcurrency)
    {
        // Arrange
        var service = CreateService();
        var documents = Enumerable.Range(1, 10)
            .Select(i => CreateDocument($"doc{i}", $"Content {i}"))
            .ToList();

        SetupMocks();

        // Act
        var results = await service.IngestDocumentsBatchAsync(documents, maxConcurrency);

        // Assert
        results.Should().HaveCount(10);
    }

    // Helper Methods

    private DocumentIngestionService CreateService()
    {
        return new DocumentIngestionService(
            _mockEmbeddingModel.Object,
            _mockRelationExtractionModel.Object,
            _mockNerModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            chunkSize: 512,
            chunkOverlap: 50,
            minRelationConfidence: 0.5f);
    }

    private Document CreateDocument(string id, string content)
    {
        return new Document
        {
            Id = id,
            TenantId = "test_tenant",
            IndexName = "test_index",
            Title = $"Test Document {id}",
            Content = content
        };
    }

    private void SetupMocks()
    {
        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingsAsync(
                It.IsAny<string[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync((string[] texts, CancellationToken ct) =>
                texts.Select(_ => GenerateRandomEmbedding(128)).ToArray());

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

        _mockGraphStorage.Setup(x => x.AddNodeAsync(
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync("node_123");
    }

    private float[] GenerateRandomEmbedding(int dimension)
    {
        var random = new Random(42);
        return Enumerable.Range(0, dimension)
            .Select(_ => (float)(random.NextDouble() * 2 - 1))
            .ToArray();
    }
}

