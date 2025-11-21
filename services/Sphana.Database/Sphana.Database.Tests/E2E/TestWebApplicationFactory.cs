using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Moq;
using Sphana.Database.Infrastructure.GraphStorage;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Models.KnowledgeGraph;
using Sphana.Database.Services;

namespace Sphana.Database.Tests.E2E;

/// <summary>
/// Custom WebApplicationFactory for E2E tests that configures the test environment
/// </summary>
public class TestWebApplicationFactory : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.ConfigureServices((context, services) =>
        {
            // Remove the original ONNX model registrations
            services.RemoveAll<IEmbeddingModel>();
            services.RemoveAll<IRelationExtractionModel>();
            services.RemoveAll<IGnnRankerModel>();
            services.RemoveAll<IVectorIndex>();
            services.RemoveAll<IGraphStorage>();
            services.RemoveAll<IDocumentIngestionService>();
            services.RemoveAll<IQueryService>();

            // Register mocked ONNX models for testing (no real ONNX files needed)
            services.AddSingleton<IEmbeddingModel>(sp =>
            {
                var mock = new Mock<IEmbeddingModel>();
                
                // Setup embedding generation to return dummy embeddings
                mock.Setup(x => x.GenerateEmbeddingsAsync(
                        It.IsAny<string[]>(),
                        It.IsAny<CancellationToken>()))
                    .ReturnsAsync((string[] texts, CancellationToken ct) =>
                    {
                        // Return random embeddings for each text
                        return texts.Select(_ => GenerateRandomEmbedding(128)).ToArray();
                    });
                
                mock.Setup(x => x.GenerateEmbeddingAsync(
                        It.IsAny<string>(),
                        It.IsAny<CancellationToken>()))
                    .ReturnsAsync(GenerateRandomEmbedding(128));
                
                return mock.Object;
            });

            services.AddSingleton<IRelationExtractionModel>(sp =>
            {
                var mock = new Mock<IRelationExtractionModel>();
                
                // Setup relation extraction to return empty list (no relations extracted)
                mock.Setup(x => x.ExtractRelationsAsync(
                        It.IsAny<string>(),
                        It.IsAny<List<ExtractedEntity>>(),
                        It.IsAny<CancellationToken>()))
                    .ReturnsAsync(new List<ExtractedRelation>());
                
                return mock.Object;
            });

            services.AddSingleton<IGnnRankerModel>(sp =>
            {
                var mock = new Mock<IGnnRankerModel>();
                
                // Setup GNN ranking to return the same subgraphs with scores
                mock.Setup(x => x.RankSubgraphsAsync(
                        It.IsAny<List<KnowledgeSubgraph>>(),
                        It.IsAny<float[]>(),
                        It.IsAny<CancellationToken>()))
                    .ReturnsAsync((List<KnowledgeSubgraph> subgraphs, float[] queryEmbedding, CancellationToken ct) =>
                    {
                        // Return the same subgraphs with random relevance scores
                        var random = new Random();
                        foreach (var sg in subgraphs)
                        {
                            sg.RelevanceScore = (float)random.NextDouble();
                        }
                        return subgraphs.OrderByDescending(sg => sg.RelevanceScore).ToList();
                    });
                
                return mock.Object;
            });

            // Register test vector index
            services.AddSingleton<IVectorIndex>(sp =>
            {
                var logger = sp.GetRequiredService<ILogger<HnswVectorIndex>>();
                return new HnswVectorIndex(
                    dimension: 128,
                    m: 16,
                    efConstruction: 200,
                    efSearch: 50,
                    distanceMetric: DistanceMetric.Cosine,
                    normalize: true,
                    logger: logger);
            });

            // Register test graph storage
            services.AddSingleton<IGraphStorage>(sp =>
            {
                var logger = sp.GetRequiredService<ILogger<PcsrGraphStorage>>();
                var tempPath = Path.Combine(Path.GetTempPath(), $"test_graph_{Guid.NewGuid()}");
                Directory.CreateDirectory(tempPath);
                return new PcsrGraphStorage(tempPath, 0.2, 4096, logger);
            });

            // Register application services with test configuration
            services.AddSingleton<IDocumentIngestionService>(sp =>
            {
                var embeddingModel = sp.GetRequiredService<IEmbeddingModel>();
                var reModel = sp.GetRequiredService<IRelationExtractionModel>();
                var vectorIndex = sp.GetRequiredService<IVectorIndex>();
                var graphStorage = sp.GetRequiredService<IGraphStorage>();
                var logger = sp.GetRequiredService<ILogger<DocumentIngestionService>>();

                return new DocumentIngestionService(
                    embeddingModel,
                    reModel,
                    vectorIndex,
                    graphStorage,
                    logger,
                    chunkSize: 512,
                    chunkOverlap: 50,
                    minRelationConfidence: 0.5f);
            });

            services.AddSingleton<IQueryService>(sp =>
            {
                var embeddingModel = sp.GetRequiredService<IEmbeddingModel>();
                var gnnModel = sp.GetRequiredService<IGnnRankerModel>();
                var vectorIndex = sp.GetRequiredService<IVectorIndex>();
                var graphStorage = sp.GetRequiredService<IGraphStorage>();
                var logger = sp.GetRequiredService<ILogger<QueryService>>();

                return new QueryService(
                    embeddingModel,
                    gnnModel,
                    vectorIndex,
                    graphStorage,
                    logger,
                    vectorSearchWeight: 0.6f,
                    graphSearchWeight: 0.4f,
                    vectorSearchTopK: 20,
                    maxSubgraphs: 10);
            });
        });

        builder.UseEnvironment("Testing");
        
        // Add detailed logging for tests
        builder.ConfigureLogging(logging =>
        {
            logging.ClearProviders();
            logging.AddConsole();
            logging.AddDebug();
            logging.SetMinimumLevel(LogLevel.Information);
        });
    }

    private static float[] GenerateRandomEmbedding(int dimension)
    {
        var random = new Random(42); // Fixed seed for reproducibility
        var embedding = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            embedding[i] = (float)(random.NextDouble() * 2 - 1); // Range: -1 to 1
        }
        
        // Normalize to unit vector
        var norm = (float)Math.Sqrt(embedding.Sum(x => x * x));
        for (int i = 0; i < dimension; i++)
        {
            embedding[i] /= norm;
        }
        
        return embedding;
    }
}

