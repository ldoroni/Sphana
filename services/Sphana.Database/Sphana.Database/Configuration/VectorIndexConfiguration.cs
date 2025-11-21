namespace Sphana.Database.Configuration;

/// <summary>
/// Configuration for vector index (HNSW/IVF)
/// </summary>
public sealed class VectorIndexConfiguration
{
    /// <summary>
    /// Type of vector index (HNSW or IVF)
    /// </summary>
    public VectorIndexType IndexType { get; set; } = VectorIndexType.Hnsw;

    /// <summary>
    /// Embedding dimension
    /// </summary>
    public int Dimension { get; set; } = 384;

    /// <summary>
    /// Distance metric for similarity search
    /// </summary>
    public DistanceMetric DistanceMetric { get; set; } = DistanceMetric.Cosine;

    /// <summary>
    /// M parameter for HNSW (number of bi-directional links)
    /// </summary>
    public int HnswM { get; set; } = 16;

    /// <summary>
    /// ef_construction parameter for HNSW
    /// </summary>
    public int HnswEfConstruction { get; set; } = 200;

    /// <summary>
    /// ef parameter for HNSW search
    /// </summary>
    public int HnswEfSearch { get; set; } = 50;

    /// <summary>
    /// Number of clusters for IVF
    /// </summary>
    public int IvfNClusters { get; set; } = 1000;

    /// <summary>
    /// Number of probes for IVF search
    /// </summary>
    public int IvfNProbes { get; set; } = 10;

    /// <summary>
    /// Use quantization for embeddings (int8)
    /// </summary>
    public bool UseQuantization { get; set; } = true;

    /// <summary>
    /// Normalize embeddings to unit vectors
    /// </summary>
    public bool NormalizeEmbeddings { get; set; } = true;

    /// <summary>
    /// Storage path for vector index
    /// </summary>
    public string StoragePath { get; set; } = "data/vector_index";

    /// <summary>
    /// Maximum number of results to return
    /// </summary>
    public int MaxResults { get; set; } = 100;
}

public enum VectorIndexType
{
    Hnsw,
    Ivf
}

public enum DistanceMetric
{
    Cosine,
    Euclidean,
    DotProduct
}

