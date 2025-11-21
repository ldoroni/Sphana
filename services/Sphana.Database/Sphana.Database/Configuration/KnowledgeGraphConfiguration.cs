namespace Sphana.Database.Configuration;

/// <summary>
/// Configuration for Knowledge Graph storage (PCSR)
/// </summary>
public sealed class KnowledgeGraphConfiguration
{
    /// <summary>
    /// Storage path for PCSR graph structure
    /// </summary>
    public string GraphStoragePath { get; set; } = "data/knowledge_graph";

    /// <summary>
    /// Storage path for Parquet property files
    /// </summary>
    public string PropertyStoragePath { get; set; } = "data/properties";

    /// <summary>
    /// Use BFS-inspired layout for disk optimization
    /// </summary>
    public bool UseBfsLayout { get; set; } = true;

    /// <summary>
    /// Slack space ratio for PCSR (e.g., 0.2 = 20% extra space for insertions)
    /// </summary>
    public double PcsrSlackRatio { get; set; } = 0.2;

    /// <summary>
    /// Block size for disk I/O (in bytes)
    /// </summary>
    public int BlockSize { get; set; } = 4096;

    /// <summary>
    /// Enable external memory support for large graphs (STXXL-style)
    /// </summary>
    public bool EnableExternalMemory { get; set; } = true;

    /// <summary>
    /// Memory limit before spilling to disk (in MB)
    /// </summary>
    public int MemoryLimitMb { get; set; } = 4096;

    /// <summary>
    /// Maximum graph traversal depth
    /// </summary>
    public int MaxTraversalDepth { get; set; } = 5;

    /// <summary>
    /// Rebalance threshold (trigger rebalancing when slack usage exceeds this ratio)
    /// </summary>
    public double RebalanceThreshold { get; set; } = 0.8;
}

