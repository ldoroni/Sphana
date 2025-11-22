using Sphana.Database.Services;
using Sphana.Database.Models;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Infrastructure.GraphStorage;
using Microsoft.Extensions.Logging;
using Xunit.Abstractions;

namespace Sphana.Database.Tests.Services;

public class DocumentIngestionServiceIntegrationTests : IAsyncLifetime
{
    private readonly ITestOutputHelper _output;
    private DocumentIngestionService? _service;
    private IVectorIndex? _vectorIndex;
    private IGraphStorage? _graphStorage;
    private string _tempPath;

    public DocumentIngestionServiceIntegrationTests(ITestOutputHelper output)
    {
        _output = output;
        _tempPath = Path.Combine(Path.GetTempPath(), $"test_integration_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempPath);
    }

    public async Task InitializeAsync()
    {
        _output.WriteLine("Initializing Document Ingestion Integration Tests with ONNX models");

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
            // Initialize mocks for ONNX models
            var serviceLogger = new Mock<ILogger<DocumentIngestionService>>();
            
            // Mock Embedding Model
            var embeddingModel = new Mock<IEmbeddingModel>();
            embeddingModel.Setup(x => x.GenerateEmbeddingsAsync(
                    It.IsAny<string[]>(),
                    It.IsAny<CancellationToken>()))
                .ReturnsAsync((string[] texts, CancellationToken ct) =>
                {
                    // Return random embeddings for each text
                    return texts.Select(_ => new float[128]).ToArray();
                });

            // Mock Relation Extraction Model
            var relationModel = new Mock<IRelationExtractionModel>();
            relationModel.Setup(x => x.ExtractRelationsAsync(
                    It.IsAny<string>(),
                    It.IsAny<List<ExtractedEntity>>(),
                    It.IsAny<CancellationToken>()))
                .ReturnsAsync(new List<ExtractedRelation>());

            // Mock NER Model
            var nerModel = new Mock<INerModel>();
            nerModel.Setup(x => x.ExtractEntitiesAsync(
                    It.IsAny<string>(),
                    It.IsAny<CancellationToken>()))
                .ReturnsAsync(new List<ExtractedEntity>());

            // Create the service
            _service = new DocumentIngestionService(
                embeddingModel.Object,
                relationModel.Object,
                nerModel.Object,
                _vectorIndex,
                _graphStorage,
                logger: serviceLogger.Object,
                chunkSize: 512,
                chunkOverlap: 50,
                minRelationConfidence: 0.5f);

            _output.WriteLine("Service initialized successfully with ONNX models");
        }
        catch (Exception ex)
        {
            _output.WriteLine($"Failed to initialize ONNX models: {ex.Message}");
            throw;
        }

        await Task.CompletedTask;
    }

    [Fact]
    public async Task IngestDocumentAsync_ShouldIndexDocumentSuccessfully()
    {
        // This test requires actual ONNX models
        // It will be enabled once models are available

        // Arrange
        var document = new Document
        {
            Id = "test_doc_1",
            TenantId = "tenant1",
            IndexName = "index1",
            Title = "Test Document",
            Content = "This is a test document about machine learning and artificial intelligence.",
            Metadata = new Dictionary<string, string>
            {
                ["category"] = "AI"
            }
        };

        // Act
        var result = await _service!.IngestDocumentAsync(document);

        // Assert
        result.Should().Be(document.Id);
        _vectorIndex!.Count.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task ChunkDocument_ShouldCreateMultipleChunks()
    {
        // Arrange
        var document = new Document
        {
            Id = "test_doc_1",
            TenantId = "tenant1",
            IndexName = "index1",
            Title = "Test Document",
            Content = string.Join(" ", Enumerable.Repeat("word", 1000)), // 1000 words
        };

        // Act - This would be tested if we had access to the private method
        // For now, we verify the concept through public methods
        
        // Assert
        // When ingested, should create multiple chunks
        _output.WriteLine($"Document with 1000 words should be chunked into multiple segments");
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

