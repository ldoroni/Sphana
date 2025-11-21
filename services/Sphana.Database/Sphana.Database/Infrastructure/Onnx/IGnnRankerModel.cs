using Sphana.Database.Models.KnowledgeGraph;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// Interface for GNN ranker model operations
/// </summary>
public interface IGnnRankerModel
{
    /// <summary>
    /// Ranks knowledge subgraphs by relevance to a query
    /// </summary>
    /// <param name="subgraphs">List of knowledge subgraphs to rank</param>
    /// <param name="queryEmbedding">Query embedding vector</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Ranked list of subgraphs with relevance scores</returns>
    Task<List<KnowledgeSubgraph>> RankSubgraphsAsync(
        List<KnowledgeSubgraph> subgraphs,
        float[] queryEmbedding,
        CancellationToken cancellationToken = default);
}

