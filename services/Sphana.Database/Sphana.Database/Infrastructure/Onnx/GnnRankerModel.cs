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
        var (nodeFeatures, edgeIndex, edgeDirections) = PrepareGraphTensors(subgraph);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("node_features", nodeFeatures),
            NamedOnnxValue.CreateFromTensor("edge_index", edgeIndex),
            NamedOnnxValue.CreateFromTensor("edge_directions", edgeDirections)
        };

        using var results = session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        // The output should be a single relevance score
        return output[0];
    }

    private (Tensor<float> NodeFeatures, Tensor<long> EdgeIndex, Tensor<long> EdgeDirections) 
        PrepareGraphTensors(KnowledgeSubgraph subgraph)
    {
        int numNodes = subgraph.Entities.Count;
        int featureDim = 384; // Embedding dimension

        // Node features: [num_nodes, feature_dim]
        var nodeFeatureData = new float[numNodes][];
        for (int i = 0; i < numNodes; i++)
        {
            var entity = subgraph.Entities[i];
            nodeFeatureData[i] = entity.Embedding ?? new float[featureDim];
        }
        var nodeFeatures = CreateTensor(nodeFeatureData, new[] { numNodes, featureDim });

        // Build entity ID to index map
        var entityIdToIndex = subgraph.Entities
            .Select((e, idx) => (e.Id, idx))
            .ToDictionary(x => x.Id, x => x.idx);

        // Build edges
        // For bidirectional GNN, we add forward (0) and backward (1) edges
        var edgesList = new List<(int Src, int Dst, long Dir)>();
        
        foreach (var relation in subgraph.Relations)
        {
            if (entityIdToIndex.TryGetValue(relation.SourceEntityId, out int sourceIdx) &&
                entityIdToIndex.TryGetValue(relation.TargetEntityId, out int targetIdx))
            {
                // Forward edge: u -> v, dir 0
                edgesList.Add((sourceIdx, targetIdx, 0));
                // Backward edge: v -> u, dir 1
                edgesList.Add((targetIdx, sourceIdx, 1));
            }
        }

        int numEdges = edgesList.Count;
        
        // Create tensors
        // edge_index: [num_edges, 2]
        var edgeIndexData = new long[Math.Max(numEdges, 1)][]; 
        var edgeDirsData = new long[Math.Max(numEdges, 1)][];

        if (numEdges == 0)
        {
            // Dummy empty edge to satisfy shape requirements if needed, 
            // but num_edges=0 might be handled by ONNX Runtime if dynamic axes allow it.
            // If dynamic axes {edges} allows 0, we can pass empty tensor.
            // However, exporter sets dynamic axes.
            // Let's try passing empty tensors if supported, or dummy if not.
            // For safety with potentially fixed or strictly typed inputs, we'll use 0-length if possible.
            // But CreateTensor helper takes arrays. 
            // Let's use empty array if numEdges is 0.
            edgeIndexData = Array.Empty<long[]>();
            edgeDirsData = Array.Empty<long[]>();
            
            // Actually, let's verify ONNX Runtime behavior. If dimension is 0, it should be fine.
            // But our CreateTensor helper creates DenseTensor.
            if (numEdges == 0)
            {
                return (
                    nodeFeatures, 
                    new DenseTensor<long>(new[] { 0, 2 }), 
                    new DenseTensor<long>(new[] { 0 })
                );
            }
        }
        else
        {
            for (int i = 0; i < numEdges; i++)
            {
                edgeIndexData[i] = new long[] { edgesList[i].Src, edgesList[i].Dst };
                edgeDirsData[i] = new long[] { edgesList[i].Dir };
            }
        }

        var edgeIndex = CreateTensor(edgeIndexData, new[] { numEdges, 2 });
        // edge_directions: [num_edges] (1D tensor)
        // Our CreateTensor Helper for long[][] creates a flattened tensor from 2D array.
        // We want [num_edges].
        // We can reuse CreateTensor but need to pass [num_edges] as dimension.
        var edgeDirections = CreateTensor(edgeDirsData, new[] { numEdges });

        return (nodeFeatures, edgeIndex, edgeDirections);
    }
}
