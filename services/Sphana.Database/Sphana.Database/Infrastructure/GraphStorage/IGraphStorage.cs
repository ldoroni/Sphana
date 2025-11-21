namespace Sphana.Database.Infrastructure.GraphStorage;

/// <summary>
/// Interface for knowledge graph storage operations
/// </summary>
public interface IGraphStorage : IDisposable
{
    // Node operations
    Task<string> AddNodeAsync(string tenantId, string indexName, string nodeData, CancellationToken cancellationToken = default);
    Task<GraphNode?> GetNodeAsync(string nodeId, CancellationToken cancellationToken = default);
    Task<bool> RemoveNodeAsync(string nodeId, CancellationToken cancellationToken = default);

    // Edge operations
    Task AddEdgeAsync(string sourceId, string targetId, string edgeData, CancellationToken cancellationToken = default);
    Task<List<GraphEdge>> GetOutgoingEdgesAsync(string nodeId, CancellationToken cancellationToken = default);
    Task<List<GraphEdge>> GetIncomingEdgesAsync(string nodeId, CancellationToken cancellationToken = default);
    Task<bool> RemoveEdgeAsync(string sourceId, string targetId, CancellationToken cancellationToken = default);

    // Traversal operations
    Task<List<GraphNode>> TraverseAsync(string startNodeId, int maxDepth, CancellationToken cancellationToken = default);
    Task<List<List<string>>> FindPathsAsync(string startNodeId, string endNodeId, int maxDepth, CancellationToken cancellationToken = default);

    // Storage operations
    Task SaveAsync(string path, CancellationToken cancellationToken = default);
    Task LoadAsync(string path, CancellationToken cancellationToken = default);
    Task CompactAsync(CancellationToken cancellationToken = default);

    // Statistics
    int NodeCount { get; }
    int EdgeCount { get; }
}

/// <summary>
/// Represents a node in the knowledge graph
/// </summary>
public sealed class GraphNode
{
    public required string Id { get; init; }
    public required string TenantId { get; init; }
    public required string IndexName { get; init; }
    public required string Data { get; init; }
    public int OutDegree { get; init; }
    public int InDegree { get; init; }
}

/// <summary>
/// Represents an edge in the knowledge graph
/// </summary>
public sealed class GraphEdge
{
    public required string SourceId { get; init; }
    public required string TargetId { get; init; }
    public required string Data { get; init; }
}

