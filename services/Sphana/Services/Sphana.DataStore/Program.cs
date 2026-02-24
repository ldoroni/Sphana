using System.Text.Json;
using System.Text.Json.Serialization;
using Prometheus;
using Sphana.DataStore.Application.Services;
using Sphana.DataStore.Domain.Interfaces;
using Sphana.DataStore.Domain.Models;
using Sphana.DataStore.Infrastructure.Configuration;
using Sphana.DataStore.Infrastructure.Persistence.Faiss;
using Sphana.DataStore.Infrastructure.Persistence.Lmdb;
using Sphana.DataStore.Infrastructure.Persistence.RocksDb;
using Sphana.DataStore.Infrastructure.Serialization;
using Sphana.DataStore.Presentation.Endpoints;
using Sphana.RequestHandler;

var builder = WebApplication.CreateBuilder(args);

// ── Configuration ──────────────────────────────────────────────────────────
var dataStoreConfiguration = builder.Configuration
    .GetSection(DataStoreConfiguration.SectionName)
    .Get<DataStoreConfiguration>() ?? new DataStoreConfiguration();

builder.Services.AddSingleton(dataStoreConfiguration);

// ── Infrastructure: RocksDB document repositories ──────────────────────────
var databasePath = Path.GetFullPath(dataStoreConfiguration.DatabasePath);

builder.Services.AddSingleton<IDocumentRepository<IndexDetails>>(serviceProvider =>
    new RocksDbDocumentRepository<IndexDetails>(
        Path.Combine(databasePath, "index_details"),
        serviceProvider.GetRequiredService<ILogger<RocksDbDocumentRepository<IndexDetails>>>(),
        AppJsonSerializerContext.Default.IndexDetails));

builder.Services.AddSingleton<IDocumentRepository<ShardDetails>>(serviceProvider =>
    new RocksDbDocumentRepository<ShardDetails>(
        Path.Combine(databasePath, "shard_details"),
        serviceProvider.GetRequiredService<ILogger<RocksDbDocumentRepository<ShardDetails>>>(),
        AppJsonSerializerContext.Default.ShardDetails));

builder.Services.AddSingleton<IDocumentRepository<EmbeddingDetails>>(serviceProvider =>
    new RocksDbDocumentRepository<EmbeddingDetails>(
        Path.Combine(databasePath, "embedding_details"),
        serviceProvider.GetRequiredService<ILogger<RocksDbDocumentRepository<EmbeddingDetails>>>(),
        AppJsonSerializerContext.Default.EmbeddingDetails));

// ── Infrastructure: concrete repositories ──────────────────────────────────
builder.Services.AddSingleton<IIndexDetailsRepository, RocksDbIndexDetailsRepository>();
builder.Services.AddSingleton<IShardDetailsRepository, RocksDbShardDetailsRepository>();
builder.Services.AddSingleton<IEmbeddingDetailsRepository, RocksDbEmbeddingDetailsRepository>();

// ── Infrastructure: LMDB blob repository ───────────────────────────────────
builder.Services.AddSingleton<IBlobRepository>(serviceProvider =>
    new LmdbBlobRepository(
        Path.Combine(databasePath, "payloads"),
        serviceProvider.GetRequiredService<ILogger<LmdbBlobRepository>>()));

// ── Infrastructure: FAISS vector repository ────────────────────────────────
builder.Services.AddSingleton<IVectorRepository, FaissVectorRepository>();

// ── Application services ───────────────────────────────────────────────────
builder.Services.AddSingleton<IIndexService, IndexService>();
builder.Services.AddSingleton<IEmbeddingService, EmbeddingService>();
builder.Services.AddSingleton<IQueryService, QueryService>();
builder.Services.AddSingleton<IPayloadService, PayloadService>();
builder.Services.AddSingleton<IEntryService, EntryService>();

// ── Presentation: JSON serialization (AOT source-generated) ────────────────
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    options.SerializerOptions.TypeInfoResolverChain.Add(AppJsonSerializerContext.Default);
});

// ── OpenAPI ────────────────────────────────────────────────────────────────
builder.Services.AddOpenApi();

// ── Build ──────────────────────────────────────────────────────────────────
var app = builder.Build();

// ── Middleware pipeline ────────────────────────────────────────────────────
app.UseManagedExceptionHandler(AppJsonSerializerContext.Default.ErrorResponse);

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

// ── Prometheus metrics ─────────────────────────────────────────────────────
app.UseHttpMetrics();
app.MapMetrics();

// ── Routing: Minimal API endpoints ─────────────────────────────────────────
app.MapIndicesEndpoints();
app.MapEmbeddingsEndpoints();
app.MapEntriesEndpoints();
app.MapPayloadsEndpoints();
app.MapQueriesEndpoints();

app.Run();