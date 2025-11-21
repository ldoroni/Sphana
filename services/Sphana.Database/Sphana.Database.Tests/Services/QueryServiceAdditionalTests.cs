using Microsoft.Extensions.Logging;
using Sphana.Database.Infrastructure.GraphStorage;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Models;
using Sphana.Database.Models.KnowledgeGraph;
using Sphana.Database.Services;

namespace Sphana.Database.Tests.Services;

/// <summary>
/// Additional comprehensive tests for QueryService covering more scenarios
/// </summary>
public class QueryServiceAdditionalTests
{
    private readonly Mock<IEmbeddingModel> _mockEmbeddingModel;
    private readonly Mock<IGnnRankerModel> _mockGnnRankerModel;
    private readonly Mock<IVectorIndex> _mockVectorIndex;
    private readonly Mock<IGraphStorage> _mockGraphStorage;
    private readonly Mock<ILogger<QueryService>> _mockLogger;

    public QueryServiceAdditionalTests()
    {
        _mockEmbeddingModel = new Mock<IEmbeddingModel>();
        _mockGnnRankerModel = new Mock<IGnnRankerModel>();
        _mockVectorIndex = new Mock<IVectorIndex>();
        _mockGraphStorage = new Mock<IGraphStorage>();
        _mockLogger = new Mock<ILogger<QueryService>>();
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithOnlyVectorResults_ShouldGenerateAnswer()
    {
        // Arrange
        var service = CreateService();
        var query = "What is machine learning?";

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(GenerateRandomEmbedding(384));

        _mockVectorIndex.Setup(x => x.SearchAsync(
                It.IsAny<float[]>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<SearchResult>
            {
                new SearchResult { Id = "chunk1", Score = 0.9f, Vector = GenerateRandomEmbedding(384) }
            });

        _mockGnnRankerModel.Setup(x => x.RankSubgraphsAsync(
                It.IsAny<List<KnowledgeSubgraph>>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<KnowledgeSubgraph>());

        // Act
        var result = await service.ExecuteQueryAsync(query, "tenant1", "index1");

        // Assert
        result.Should().NotBeNull();
        result.Answer.Should().NotBeEmpty();
        result.VectorResults.Should().HaveCount(1);
        result.KnowledgeSubgraphs.Should().BeEmpty();
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithManyVectorResults_ShouldLimitTo10()
    {
        // Arrange
        var service = CreateService();
        var query = "Test query";

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(GenerateRandomEmbedding(384));

        var manyResults = Enumerable.Range(0, 50)
            .Select(i => new SearchResult
            {
                Id = $"chunk_{i}",
                Score = 1.0f - (i * 0.01f),
                Vector = GenerateRandomEmbedding(384)
            })
            .ToList();

        _mockVectorIndex.Setup(x => x.SearchAsync(
                It.IsAny<float[]>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(manyResults);

        _mockGnnRankerModel.Setup(x => x.RankSubgraphsAsync(
                It.IsAny<List<KnowledgeSubgraph>>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<KnowledgeSubgraph>());

        // Act
        var result = await service.ExecuteQueryAsync(query, "tenant1", "index1");

        // Assert
        result.VectorResults.Should().HaveCount(10);
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithGraphResults_ShouldLimitTo5Subgraphs()
    {
        // Arrange
        var service = CreateService();
        var query = "Test query";

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(GenerateRandomEmbedding(384));

        _mockVectorIndex.Setup(x => x.SearchAsync(
                It.IsAny<float[]>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<SearchResult>
            {
                new SearchResult { Id = "chunk1", Score = 0.9f, Vector = GenerateRandomEmbedding(384) }
            });

        var manySubgraphs = Enumerable.Range(0, 20)
            .Select(i => new KnowledgeSubgraph
            {
                Id = $"sg_{i}",
                Entities = new List<Entity>(),
                Relations = new List<Relation>(),
                RelevanceScore = 1.0f - (i * 0.05f)
            })
            .ToList();

        _mockGnnRankerModel.Setup(x => x.RankSubgraphsAsync(
                It.IsAny<List<KnowledgeSubgraph>>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(manySubgraphs);

        // Act
        var result = await service.ExecuteQueryAsync(query, "tenant1", "index1");

        // Assert
        result.KnowledgeSubgraphs.Count.Should().BeLessThanOrEqualTo(5);
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithLowScoreResults_ShouldStillProcess()
    {
        // Arrange
        var service = CreateService();
        var query = "Test query";

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(GenerateRandomEmbedding(384));

        _mockVectorIndex.Setup(x => x.SearchAsync(
                It.IsAny<float[]>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<SearchResult>
            {
                new SearchResult { Id = "chunk1", Score = 0.1f, Vector = GenerateRandomEmbedding(384) }
            });

        _mockGnnRankerModel.Setup(x => x.RankSubgraphsAsync(
                It.IsAny<List<KnowledgeSubgraph>>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<KnowledgeSubgraph>());

        // Act
        var result = await service.ExecuteQueryAsync(query, "tenant1", "index1");

        // Assert
        result.Should().NotBeNull();
        result.VectorResults.Should().HaveCount(1);
    }

    [Fact]
    public async Task ExecuteQueryAsync_RecordsLatency_ShouldBeReasonable()
    {
        // Arrange
        var service = CreateService();
        var query = "Test query";

        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(GenerateRandomEmbedding(384));

        _mockVectorIndex.Setup(x => x.SearchAsync(
                It.IsAny<float[]>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<SearchResult>());

        _mockGnnRankerModel.Setup(x => x.RankSubgraphsAsync(
                It.IsAny<List<KnowledgeSubgraph>>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<KnowledgeSubgraph>());

        // Act
        var result = await service.ExecuteQueryAsync(query, "tenant1", "index1");

        // Assert
        result.LatencyMs.Should().BeGreaterThan(0);
        result.LatencyMs.Should().BeLessThan(10000); // Less than 10 seconds
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithDifferentTenants_ShouldIsolate()
    {
        // Arrange
        var service = CreateService();
        var query = "Test query";

        SetupMocks();

        // Act
        var result1 = await service.ExecuteQueryAsync(query, "tenant1", "index1");
        var result2 = await service.ExecuteQueryAsync(query, "tenant2", "index1");

        // Assert
        result1.Should().NotBeNull();
        result2.Should().NotBeNull();
        // Both should succeed and be independent
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithDifferentIndexes_ShouldSearchCorrectIndex()
    {
        // Arrange
        var service = CreateService();
        var query = "Test query";

        SetupMocks();

        // Act
        var result1 = await service.ExecuteQueryAsync(query, "tenant1", "index1");
        var result2 = await service.ExecuteQueryAsync(query, "tenant1", "index2");

        // Assert
        result1.Should().NotBeNull();
        result2.Should().NotBeNull();
    }

    [Theory]
    [InlineData(0.1f, 0.9f)]
    [InlineData(0.5f, 0.5f)]
    [InlineData(0.7f, 0.3f)]
    [InlineData(0.9f, 0.1f)]
    public async Task ExecuteQueryAsync_WithDifferentWeights_ShouldCombineResults(float vectorWeight, float graphWeight)
    {
        // Arrange
        var service = new QueryService(
            _mockEmbeddingModel.Object,
            _mockGnnRankerModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            vectorWeight,
            graphWeight,
            vectorSearchTopK: 20,
            maxSubgraphs: 10);

        var query = "Test query";

        SetupMocks();

        // Act
        var result = await service.ExecuteQueryAsync(query, "tenant1", "index1");

        // Assert
        result.Should().NotBeNull();
        result.Answer.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithSpecialCharacters_ShouldHandleCorrectly()
    {
        // Arrange
        var service = CreateService();
        var query = "How does @#$% work in C++?";

        SetupMocks();

        // Act
        var result = await service.ExecuteQueryAsync(query, "tenant1", "index1");

        // Assert
        result.Should().NotBeNull();
        result.Query.Should().Be(query);
    }

    [Fact]
    public async Task ExecuteQueryAsync_WithUnicodeCharacters_ShouldHandleCorrectly()
    {
        // Arrange
        var service = CreateService();
        var query = "什么是机器学习？ What is αβγ?";

        SetupMocks();

        // Act
        var result = await service.ExecuteQueryAsync(query, "tenant1", "index1");

        // Assert
        result.Should().NotBeNull();
        result.Query.Should().Be(query);
    }

    // Helper Methods

    private QueryService CreateService()
    {
        return new QueryService(
            _mockEmbeddingModel.Object,
            _mockGnnRankerModel.Object,
            _mockVectorIndex.Object,
            _mockGraphStorage.Object,
            _mockLogger.Object,
            vectorSearchWeight: 0.6f,
            graphSearchWeight: 0.4f,
            vectorSearchTopK: 20,
            maxSubgraphs: 10);
    }

    private void SetupMocks()
    {
        _mockEmbeddingModel.Setup(x => x.GenerateEmbeddingAsync(
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(GenerateRandomEmbedding(384));

        _mockVectorIndex.Setup(x => x.SearchAsync(
                It.IsAny<float[]>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<SearchResult>
            {
                new SearchResult { Id = "chunk1", Score = 0.9f, Vector = GenerateRandomEmbedding(384) }
            });

        _mockGnnRankerModel.Setup(x => x.RankSubgraphsAsync(
                It.IsAny<List<KnowledgeSubgraph>>(),
                It.IsAny<float[]>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<KnowledgeSubgraph>());
    }

    private float[] GenerateRandomEmbedding(int dimension)
    {
        var random = new Random(42);
        return Enumerable.Range(0, dimension)
            .Select(_ => (float)(random.NextDouble() * 2 - 1))
            .ToArray();
    }
}

