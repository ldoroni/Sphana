using BERTTokenizers.Base;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;
using Sphana.Database.Configuration;
using Sphana.Database.Controllers;
using Sphana.Database.Infrastructure.GraphStorage;
using Sphana.Database.Infrastructure.Logging;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.Tokenizers;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Services;

var builder = WebApplication.CreateBuilder(args);

// Configure custom logging format
builder.Logging.ClearProviders();
builder.Logging.AddConsole(options =>
{
    options.FormatterName = "sphana";
});
builder.Logging.AddConsoleFormatter<SphanaConsoleFormatter, Microsoft.Extensions.Logging.Console.ConsoleFormatterOptions>();

// Add configuration
var sphanaConfig = builder.Configuration.GetSection("Sphana").Get<SphanaConfiguration>() 
    ?? new SphanaConfiguration();
builder.Services.AddSingleton(sphanaConfig);

// Add OpenTelemetry
builder.Services.AddOpenTelemetry()
    .WithMetrics(metrics =>
    {
        metrics
            .SetResourceBuilder(ResourceBuilder.CreateDefault().AddService("Sphana.Database"))
            .AddAspNetCoreInstrumentation()
            .AddMeter("System.Runtime") // Runtime instrumentation
            .AddPrometheusExporter();
    })
    .WithTracing(tracing =>
    {
        tracing
            .SetResourceBuilder(ResourceBuilder.CreateDefault().AddService("Sphana.Database"))
            .AddAspNetCoreInstrumentation()
            .AddSource("Grpc.*"); // gRPC instrumentation
    });

// Add gRPC services
builder.Services.AddGrpc(options =>
{
    options.MaxReceiveMessageSize = 16 * 1024 * 1024; // 16 MB
    options.MaxSendMessageSize = 16 * 1024 * 1024; // 16 MB
});

// Add in-memory caching
builder.Services.AddMemoryCache();

// Add health checks
builder.Services.AddHealthChecks()
    .AddCheck<SphanaDatabaseHealthCheck>("sphana_database");

// Register ONNX models as singletons
builder.Services.AddSingleton<UncasedTokenizer>(sp =>
{
    return new CustomBertUncasedBaseTokenizer(
        sphanaConfig.Models.VocabulariesPath
    );
});

builder.Services.AddSingleton<IEmbeddingModel>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<EmbeddingModel>>();
    var tokenizer = sp.GetRequiredService<UncasedTokenizer>();
    return new EmbeddingModel(
        sphanaConfig.Models.EmbeddingModelPath,
        sphanaConfig.Models.EmbeddingDimension,
        sphanaConfig.Models.UseGpu,
        sphanaConfig.Models.GpuDeviceId,
        maxPoolSize: 4,
        sphanaConfig.Models.BatchSize,
        sphanaConfig.Models.MaxBatchWaitMs,
        tokenizer,
        logger);
});

builder.Services.AddSingleton<IRelationExtractionModel>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<RelationExtractionModel>>();
    var tokenizer = sp.GetRequiredService<UncasedTokenizer>();
    return new RelationExtractionModel(
        sphanaConfig.Models.RelationExtractionModelPath,
        sphanaConfig.Models.UseGpu,
        sphanaConfig.Models.GpuDeviceId,
        maxPoolSize: 2,
        tokenizer,
        logger);
});

builder.Services.AddSingleton<INerModel>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<NerModel>>();
    var tokenizer = sp.GetRequiredService<UncasedTokenizer>();
    return new NerModel(
        sphanaConfig.Models.NerModelPath,
        sphanaConfig.Models.UseGpu,
        sphanaConfig.Models.GpuDeviceId,
        maxPoolSize: 2,
        tokenizer,
        logger);
});

builder.Services.AddSingleton<IGnnRankerModel>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<GnnRankerModel>>();
    return new GnnRankerModel(
        sphanaConfig.Models.GnnRankerModelPath,
        sphanaConfig.Models.UseGpu,
        sphanaConfig.Models.GpuDeviceId,
        maxPoolSize: 2,
        logger);
});

builder.Services.AddSingleton<ILlmGeneratorModel>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<LlmGeneratorModel>>();
    var debugEnabled = sphanaConfig.Debug.EnableVerboseLogging && sphanaConfig.Debug.LogTokenizedText;
    return new LlmGeneratorModel(
        sphanaConfig.Models.LlmGeneratorModelPath,
        sphanaConfig.Models.LlmTokenizerPath,
        sphanaConfig.Models.UseGpu,
        sphanaConfig.Models.GpuDeviceId,
        maxPoolSize: 1, // LLMs use a lot of VRAM
        debugEnabled,
        logger);
});

// Register vector index
builder.Services.AddSingleton<IVectorIndex>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<HnswVectorIndex>>();
    var index = new HnswVectorIndex(
        dimension: sphanaConfig.VectorIndex.Dimension,
        m: sphanaConfig.VectorIndex.HnswM,
        efConstruction: sphanaConfig.VectorIndex.HnswEfConstruction,
        efSearch: sphanaConfig.VectorIndex.HnswEfSearch,
        distanceMetric: (Sphana.Database.Infrastructure.VectorIndex.DistanceMetric)sphanaConfig.VectorIndex.DistanceMetric,
        normalize: sphanaConfig.VectorIndex.NormalizeEmbeddings,
        logger: logger);

    // Try to load existing index
    var indexPath = Path.Combine(sphanaConfig.VectorIndex.StoragePath, "index.bin");
    if (File.Exists(indexPath))
    {
        try
        {
            index.LoadAsync(indexPath).Wait();
            logger.LogInformation("Loaded existing vector index from {Path}", indexPath);
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Failed to load existing vector index, starting fresh");
        }
    }

    return index;
});

// Register graph storage
builder.Services.AddSingleton<IGraphStorage>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<PcsrGraphStorage>>();
    var storage = new PcsrGraphStorage(
        sphanaConfig.KnowledgeGraph.GraphStoragePath,
        sphanaConfig.KnowledgeGraph.PcsrSlackRatio,
        sphanaConfig.KnowledgeGraph.BlockSize,
        logger);

    // Try to load existing graph
    var graphPath = "graph.bin";
    if (File.Exists(Path.Combine(sphanaConfig.KnowledgeGraph.GraphStoragePath, graphPath)))
    {
        try
        {
            storage.LoadAsync(graphPath).Wait();
            logger.LogInformation("Loaded existing knowledge graph from {Path}", graphPath);
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Failed to load existing knowledge graph, starting fresh");
        }
    }

    return storage;
});

// Register application services
builder.Services.AddSingleton<IDocumentIngestionService>(sp =>
{
    var embeddingModel = sp.GetRequiredService<IEmbeddingModel>();
    var reModel = sp.GetRequiredService<IRelationExtractionModel>();
    var nerModel = sp.GetRequiredService<INerModel>();
    var vectorIndex = sp.GetRequiredService<IVectorIndex>();
    var graphStorage = sp.GetRequiredService<IGraphStorage>();
    var logger = sp.GetRequiredService<ILogger<DocumentIngestionService>>();

    var debugEnabled = sphanaConfig.Debug.EnableVerboseLogging && sphanaConfig.Debug.LogIngestionData;

    return new DocumentIngestionService(
        embeddingModel,
        reModel,
        nerModel,
        vectorIndex,
        graphStorage,
        logger,
        sphanaConfig.Ingestion.ChunkSize,
        sphanaConfig.Ingestion.ChunkOverlap,
        sphanaConfig.Ingestion.MinRelationConfidence,
        debugEnabled);
});

builder.Services.AddSingleton<IQueryService>(sp =>
{
    var embeddingModel = sp.GetRequiredService<IEmbeddingModel>();
    var gnnModel = sp.GetRequiredService<IGnnRankerModel>();
    var llmModel = sp.GetRequiredService<ILlmGeneratorModel>();
    var nerModel = sp.GetRequiredService<INerModel>();
    var vectorIndex = sp.GetRequiredService<IVectorIndex>();
    var graphStorage = sp.GetRequiredService<IGraphStorage>();
    var logger = sp.GetRequiredService<ILogger<QueryService>>();

    return new QueryService(
        embeddingModel,
        gnnModel,
        llmModel,
        nerModel,
        vectorIndex,
        graphStorage,
        logger,
        sphanaConfig.Query.VectorSearchWeight,
        sphanaConfig.Query.GraphSearchWeight,
        sphanaConfig.Query.VectorSearchTopK,
        sphanaConfig.Query.MaxSubgraphs,
        sphanaConfig.Query.MaxGenerationTokens);
});

// Add health checks
builder.Services.AddHealthChecks();

var app = builder.Build();

// Configure the HTTP request pipeline
app.MapGrpcService<SphanaDatabaseGrpcController>();
app.MapHealthChecks("/health");
app.MapPrometheusScrapingEndpoint();

app.MapGet("/", () => 
    "Sphana Database - Neural RAG Database System. " +
    "Communication with gRPC endpoints must be made through a gRPC client.");

// Graceful shutdown: save indexes
var lifetime = app.Services.GetRequiredService<IHostApplicationLifetime>();
lifetime.ApplicationStopping.Register(() =>
{
    var logger = app.Services.GetRequiredService<ILogger<Program>>();
    logger.LogInformation("Application stopping, saving indexes...");

    try
    {
        var vectorIndex = app.Services.GetRequiredService<IVectorIndex>();
        var indexPath = Path.Combine(sphanaConfig.VectorIndex.StoragePath, "index.bin");
        vectorIndex.SaveAsync(indexPath).Wait();
        logger.LogInformation("Vector index saved");

        var graphStorage = app.Services.GetRequiredService<IGraphStorage>();
        graphStorage.SaveAsync("graph.bin").Wait();
        logger.LogInformation("Knowledge graph saved");
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Error saving indexes during shutdown");
    }
});

app.Run();
