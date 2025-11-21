using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Sphana.Database.Models.KnowledgeGraph;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// ONNX model wrapper for Graph Neural Network (GNN) ranking
/// Uses Bi-directional Gated Graph Sequence Neural Network (GGNN) architecture
/// </summary>
public sealed class GnnRankerModel : OnnxModelBase, IGnnRankerModel
{
    public GnnRankerModel(
        string modelPath,
        bool useGpu,
        int gpuDeviceId,
        int maxPoolSize,
        ILogger<GnnRankerModel> logger)
        : base(modelPath, useGpu, gpuDeviceId, maxPoolSize, logger)
    {
    }

    /// <summary>
    /// Rank knowledge subgraphs by relevance using the GNN model (interface implementation)
    /// </summary>
    async Task<List<KnowledgeSubgraph>> IGnnRankerModel.RankSubgraphsAsync(
        List<KnowledgeSubgraph> subgraphs,
        float[] queryEmbedding,
        CancellationToken cancellationToken)
    {
        if (subgraphs == null || subgraphs.Count == 0)
        {
            return new List<KnowledgeSubgraph>();
        }

        // Use empty string for query since we have queryEmbedding
        var rankedResults = await RankSubgraphsAsync(subgraphs, string.Empty, cancellationToken);
        
        // Update relevance scores and return ordered subgraphs
        var result = new List<KnowledgeSubgraph>();
        foreach (var (subgraph, score) in rankedResults)
        {
            subgraph.RelevanceScore = score;
            result.Add(subgraph);
        }
        
        return result;
    }

    /// <summary>
    /// Rank knowledge subgraphs by relevance using the GNN model
    /// </summary>
    public async Task<List<(KnowledgeSubgraph Subgraph, float Score)>> RankSubgraphsAsync(
        List<KnowledgeSubgraph> subgraphs,
        string query,
        CancellationToken cancellationToken = default)
    {
        if (subgraphs == null || subgraphs.Count == 0)
        {
            return new List<(KnowledgeSubgraph, float)>();
        }

        var session = await AcquireSessionAsync(cancellationToken);
        try
        {
            var scores = new float[subgraphs.Count];

            for (int i = 0; i < subgraphs.Count; i++)
            {
                scores[i] = await ScoreSubgraphAsync(session, subgraphs[i], query, cancellationToken);
            }

            var rankedResults = subgraphs
                .Select((sg, idx) => (Subgraph: sg, Score: scores[idx]))
                .OrderByDescending(x => x.Score)
                .ToList();

            return rankedResults;
        }
        finally
        {
            ReleaseSession(session);
        }
    }

    private async Task<float> ScoreSubgraphAsync(
        InferenceSession session,
        KnowledgeSubgraph subgraph,
        string query,
        CancellationToken cancellationToken)
    {
        // Prepare graph tensors
        var (nodeFeatures, adjacencyMatrix, edgeFeatures) = PrepareGraphTensors(subgraph);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("node_features", nodeFeatures),
            NamedOnnxValue.CreateFromTensor("adjacency_matrix", adjacencyMatrix),
            NamedOnnxValue.CreateFromTensor("edge_features", edgeFeatures)
        };

        using var results = session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        // The output should be a single relevance score
        return output[0];
    }

    private (Tensor<float> NodeFeatures, Tensor<float> AdjacencyMatrix, Tensor<float> EdgeFeatures) 
        PrepareGraphTensors(KnowledgeSubgraph subgraph)
    {
        int numNodes = subgraph.Entities.Count;
        int numEdges = subgraph.Relations.Count;
        int featureDim = 384; // Embedding dimension

        // Node features: [num_nodes, feature_dim]
        var nodeFeatureData = new float[numNodes][];
        for (int i = 0; i < numNodes; i++)
        {
            var entity = subgraph.Entities[i];
            nodeFeatureData[i] = entity.Embedding ?? new float[featureDim];
        }
        var nodeFeatures = CreateTensor(nodeFeatureData, new[] { numNodes, featureDim });

        // Adjacency matrix: [num_nodes, num_nodes]
        // For simplicity, create a dense matrix (in practice, use sparse representation)
        var adjData = new float[numNodes][];
        for (int i = 0; i < numNodes; i++)
        {
            adjData[i] = new float[numNodes];
        }

        // Build entity ID to index map
        var entityIdToIndex = subgraph.Entities
            .Select((e, idx) => (e.Id, idx))
            .ToDictionary(x => x.Id, x => x.idx);

        // Fill adjacency matrix
        foreach (var relation in subgraph.Relations)
        {
            if (entityIdToIndex.TryGetValue(relation.SourceEntityId, out int sourceIdx) &&
                entityIdToIndex.TryGetValue(relation.TargetEntityId, out int targetIdx))
            {
                adjData[sourceIdx][targetIdx] = 1.0f;
                adjData[targetIdx][sourceIdx] = 1.0f; // Undirected graph
            }
        }
        var adjacencyMatrix = CreateTensor(adjData, new[] { numNodes, numNodes });

        // Edge features: [num_edges, feature_dim]
        // Use relation type embeddings (simplified)
        var edgeFeatureData = new float[Math.Max(numEdges, 1)][];
        for (int i = 0; i < numEdges; i++)
        {
            edgeFeatureData[i] = new float[featureDim]; // Placeholder
        }
        if (numEdges == 0)
        {
            edgeFeatureData[0] = new float[featureDim];
        }
        var edgeFeatures = CreateTensor(edgeFeatureData, new[] { Math.Max(numEdges, 1), featureDim });

        return (nodeFeatures, adjacencyMatrix, edgeFeatures);
    }
}

