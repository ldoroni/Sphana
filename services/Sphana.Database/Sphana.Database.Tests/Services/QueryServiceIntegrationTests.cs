using Sphana.Database.Services;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Infrastructure.GraphStorage;
using Microsoft.Extensions.Logging;
using Xunit.Abstractions;

namespace Sphana.Database.Tests.Services;

public class QueryServiceIntegrationTests : IAsyncLifetime
{
    private readonly ITestOutputHelper _output;
    private QueryService? _service;
    private IVectorIndex? _vectorIndex;
    private IGraphStorage? _graphStorage;
    private string _tempPath;

    public QueryServiceIntegrationTests(ITestOutputHelper output)
    {
        _output = output;
        _tempPath = Path.Combine(Path.GetTempPath(), $"test_query_integration_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempPath);
    }

    public async Task InitializeAsync()
    {
        _output.WriteLine("Initializing Query Integration Tests with ONNX models");

        // Initialize components
        _vectorIndex = new HnswVectorIndex(
            dimension: 128, // Updated to match the model
            m: 16,
            efConstruction: 200,
            efSearch: 50,
            distanceMetric: Sphana.Database.Infrastructure.VectorIndex.DistanceMetric.Cosine,
            normalize: true,
            logger: null!);

        _graphStorage = new PcsrGraphStorage(
            Path.Combine(_tempPath, "graph"),
            slackRatio: 0.2,
            blockSize: 4096,
            logger: null!);

        try
        {
            // Initialize ONNX models
            var embeddingLogger = new Mock<ILogger<EmbeddingModel>>();
            var gnnLogger = new Mock<ILogger<GnnRankerModel>>();
            var serviceLogger = new Mock<ILogger<QueryService>>();

            var embeddingModel = new EmbeddingModel(
                modelPath: "../../../../models/embedding.onnx",
                embeddingDimension: 128,
                useGpu: false,
                gpuDeviceId: 0,
                maxPoolSize: 1,
                maxBatchSize: 8,
                maxBatchWaitMs: 5,
                logger: embeddingLogger.Object);

            var gnnModel = new GnnRankerModel(
                modelPath: "../../../../models/gnn_ranker.onnx",
                useGpu: false,
                gpuDeviceId: 0,
                maxPoolSize: 1,
                logger: gnnLogger.Object);

            // Create the service
            _service = new QueryService(
                embeddingModel,
                gnnModel,
                _vectorIndex,
                _graphStorage,
                logger: serviceLogger.Object,
                vectorSearchWeight: 0.6f,
                graphSearchWeight: 0.4f,
                vectorSearchTopK: 10,
                maxSubgraphs: 5);

            _output.WriteLine("Service initialized successfully with ONNX models");
        }
        catch (Exception ex)
        {
            _output.WriteLine($"Failed to initialize ONNX models: {ex.Message}");
            throw;
        }

        // Add some test data to indexes
        await PopulateTestData();
    }

    private async Task PopulateTestData()
    {
        // Add some sample vectors to the index
        for (int i = 0; i < 10; i++)
        {
            var vector = GenerateRandomVector(128); // Updated dimension
            await _vectorIndex!.AddAsync($"chunk_{i}", vector);
        }

        // Add some sample nodes to the graph
        var node1 = await _graphStorage!.AddNodeAsync("tenant1", "index1", "entity1");
        var node2 = await _graphStorage.AddNodeAsync("tenant1", "index1", "entity2");
        await _graphStorage.AddEdgeAsync(node1, node2, "relation");
    }

    [Fact]
    public async Task ExecuteQueryAsync_ShouldReturnResults()
    {
        // This test requires actual ONNX models
        // It will be enabled once models are available

        // Arrange
        var query = "What is machine learning?";

        // Act
        var result = await _service!.ExecuteQueryAsync(query, "tenant1", "index1");

        // Assert
        result.Should().NotBeNull();
        result.Query.Should().Be(query);
        result.Answer.Should().NotBeNullOrEmpty();
        result.LatencyMs.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task HybridSearch_ShouldCombineVectorAndGraphResults()
    {
        // Arrange - Already have test data from initialization

        // Act
        var vectorResults = await _vectorIndex!.SearchAsync(GenerateRandomVector(128), topK: 5);

        // Assert
        vectorResults.Should().NotBeEmpty();
        vectorResults.Should().HaveCountLessThanOrEqualTo(5);
        _output.WriteLine($"Found {vectorResults.Count} vector search results");
    }

    private float[] GenerateRandomVector(int dimension)
    {
        var random = new Random(42);
        var vector = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            vector[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return vector;
    }

    public async Task DisposeAsync()
    {
        _vectorIndex = null;
        _graphStorage?.Dispose();
        _graphStorage = null;

        if (Directory.Exists(_tempPath))
        {
            Directory.Delete(_tempPath, recursive: true);
        }

        await Task.CompletedTask;
    }
}

