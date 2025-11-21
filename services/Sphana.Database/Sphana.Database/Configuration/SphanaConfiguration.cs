namespace Sphana.Database.Configuration;

/// <summary>
/// Main configuration for Sphana Database
/// </summary>
public sealed class SphanaConfiguration
{
    /// <summary>
    /// ONNX model configuration
    /// </summary>
    public OnnxModelConfiguration Models { get; set; } = new();

    /// <summary>
    /// Vector index configuration
    /// </summary>
    public VectorIndexConfiguration VectorIndex { get; set; } = new();

    /// <summary>
    /// Knowledge graph configuration
    /// </summary>
    public KnowledgeGraphConfiguration KnowledgeGraph { get; set; } = new();

    /// <summary>
    /// Cache configuration
    /// </summary>
    public CacheConfiguration Cache { get; set; } = new();

    /// <summary>
    /// Ingestion pipeline configuration
    /// </summary>
    public IngestionConfiguration Ingestion { get; set; } = new();

    /// <summary>
    /// Query pipeline configuration
    /// </summary>
    public QueryConfiguration Query { get; set; } = new();
}

/// <summary>
/// Cache configuration
/// </summary>
public sealed class CacheConfiguration
{
    /// <summary>
    /// In-memory cache size limit (in MB)
    /// </summary>
    public int InMemoryCacheSizeMb { get; set; } = 1024;

    /// <summary>
    /// Cache expiration time in minutes
    /// </summary>
    public int ExpirationMinutes { get; set; } = 60;

    /// <summary>
    /// Cache embedding results
    /// </summary>
    public bool CacheEmbeddings { get; set; } = true;

    /// <summary>
    /// Cache subgraph results
    /// </summary>
    public bool CacheSubgraphs { get; set; } = true;

    /// <summary>
    /// Cache GNN outputs
    /// </summary>
    public bool CacheGnnOutputs { get; set; } = true;
}

/// <summary>
/// Ingestion pipeline configuration
/// </summary>
public sealed class IngestionConfiguration
{
    /// <summary>
    /// Chunk size (number of tokens)
    /// </summary>
    public int ChunkSize { get; set; } = 512;

    /// <summary>
    /// Chunk overlap (number of tokens)
    /// </summary>
    public int ChunkOverlap { get; set; } = 50;

    /// <summary>
    /// Maximum concurrent ingestion operations
    /// </summary>
    public int MaxConcurrency { get; set; } = 10;

    /// <summary>
    /// Batch size for bulk operations
    /// </summary>
    public int BatchSize { get; set; } = 100;

    /// <summary>
    /// Enable relation extraction
    /// </summary>
    public bool EnableRelationExtraction { get; set; } = true;

    /// <summary>
    /// Minimum confidence threshold for relation extraction
    /// </summary>
    public float MinRelationConfidence { get; set; } = 0.5f;
}

/// <summary>
/// Query pipeline configuration
/// </summary>
public sealed class QueryConfiguration
{
    /// <summary>
    /// Target p95 latency in milliseconds
    /// </summary>
    public int TargetP95LatencyMs { get; set; } = 50;

    /// <summary>
    /// Number of vector search results to retrieve
    /// </summary>
    public int VectorSearchTopK { get; set; } = 20;

    /// <summary>
    /// Number of subgraphs to consider
    /// </summary>
    public int MaxSubgraphs { get; set; } = 10;

    /// <summary>
    /// Enable GNN reranking
    /// </summary>
    public bool EnableGnnReranking { get; set; } = true;

    /// <summary>
    /// Enable LLM answer generation
    /// </summary>
    public bool EnableLlmGeneration { get; set; } = true;

    /// <summary>
    /// Maximum tokens for LLM generation
    /// </summary>
    public int MaxGenerationTokens { get; set; } = 512;

    /// <summary>
    /// Hybrid retrieval: weight for vector search (0-1)
    /// </summary>
    public float VectorSearchWeight { get; set; } = 0.6f;

    /// <summary>
    /// Hybrid retrieval: weight for graph search (0-1)
    /// </summary>
    public float GraphSearchWeight { get; set; } = 0.4f;
}

