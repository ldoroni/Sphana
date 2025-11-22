using Sphana.Database.Models;
using Sphana.Database.Models.KnowledgeGraph;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Infrastructure.GraphStorage;
using System.Text.Json;
using System.Text;

namespace Sphana.Database.Services;

/// <summary>
/// Service for querying documents using hybrid vector + graph retrieval
/// </summary>
public sealed class QueryService : IQueryService
{
    private readonly IEmbeddingModel _embeddingModel;
    private readonly IGnnRankerModel _gnnRankerModel;
    private readonly ILlmGeneratorModel _llmGeneratorModel;
    private readonly INerModel _nerModel;
    private readonly IVectorIndex _vectorIndex;
    private readonly IGraphStorage _graphStorage;
    private readonly ILogger<QueryService> _logger;
    private readonly float _vectorSearchWeight;
    private readonly float _graphSearchWeight;
    private readonly int _vectorSearchTopK;
    private readonly int _maxSubgraphs;
    private readonly int _maxGenerationTokens;

    public QueryService(
        IEmbeddingModel embeddingModel,
        IGnnRankerModel gnnRankerModel,
        ILlmGeneratorModel llmGeneratorModel,
        INerModel nerModel,
        IVectorIndex vectorIndex,
        IGraphStorage graphStorage,
        ILogger<QueryService> logger,
        float vectorSearchWeight = 0.6f,
        float graphSearchWeight = 0.4f,
        int vectorSearchTopK = 20,
        int maxSubgraphs = 10,
        int maxGenerationTokens = 512)
    {
        _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
        _gnnRankerModel = gnnRankerModel ?? throw new ArgumentNullException(nameof(gnnRankerModel));
        _llmGeneratorModel = llmGeneratorModel ?? throw new ArgumentNullException(nameof(llmGeneratorModel));
        _nerModel = nerModel ?? throw new ArgumentNullException(nameof(nerModel));
        _vectorIndex = vectorIndex ?? throw new ArgumentNullException(nameof(vectorIndex));
        _graphStorage = graphStorage ?? throw new ArgumentNullException(nameof(graphStorage));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _vectorSearchWeight = vectorSearchWeight;
        _graphSearchWeight = graphSearchWeight;
        _vectorSearchTopK = vectorSearchTopK;
        _maxSubgraphs = maxSubgraphs;
        _maxGenerationTokens = maxGenerationTokens;
    }

    /// <summary>
    /// Execute a hybrid query combining vector and graph search
    /// </summary>
    public async Task<QueryResult> ExecuteQueryAsync(
        string query,
        string tenantId,
        string indexName,
        CancellationToken cancellationToken = default)
    {
        // Validate inputs
        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Query cannot be null or empty", nameof(query));
        }

        if (string.IsNullOrWhiteSpace(tenantId))
        {
            throw new ArgumentException("Tenant ID cannot be null or empty", nameof(tenantId));
        }

        if (string.IsNullOrWhiteSpace(indexName))
        {
            throw new ArgumentException("Index name cannot be null or empty", nameof(indexName));
        }

        _logger.LogInformation("Executing query: {Query} for tenant {TenantId}, index {IndexName}",
            query, tenantId, indexName);

        var startTime = DateTime.UtcNow;

        try
        {
            // Step 1: Generate query embedding
            var queryEmbedding = await _embeddingModel.GenerateEmbeddingAsync(query, cancellationToken);
            _logger.LogDebug("Generated query embedding");

            // Step 2: Vector search
            var vectorResults = await _vectorIndex.SearchAsync(queryEmbedding, _vectorSearchTopK, cancellationToken);
            _logger.LogDebug("Vector search returned {Count} results", vectorResults.Count);

            // Step 3: Extract entities from query
            var queryEntities = await ExtractQueryEntitiesAsync(query, cancellationToken);
            _logger.LogDebug("Extracted {Count} entities from query", queryEntities.Count);

            // Step 4: Graph search - find relevant subgraphs
            var subgraphs = await ExtractKnowledgeSubgraphsAsync(
                queryEntities, 
                vectorResults, 
                cancellationToken);
            _logger.LogDebug("Extracted {Count} knowledge subgraphs", subgraphs.Count);

            // Step 5: GNN reranking of subgraphs
            var rankedSubgraphsList = await _gnnRankerModel.RankSubgraphsAsync(
                subgraphs, 
                queryEmbedding, 
                cancellationToken);
            _logger.LogDebug("Ranked {Count} subgraphs using GNN", rankedSubgraphsList.Count);
            
            // Convert to tuple list for CombineResults
            var rankedSubgraphs = rankedSubgraphsList.Select(sg => (sg, sg.RelevanceScore)).ToList();

            // Step 6: Combine vector and graph results
            var combinedResults = CombineResults(vectorResults, rankedSubgraphs);

            // Step 7: Generate final answer (LLM)
            var answer = await GenerateAnswerAsync(query, combinedResults, cancellationToken);

            var latency = (DateTime.UtcNow - startTime).TotalMilliseconds;
            _logger.LogInformation("Query completed in {Latency}ms", latency);

            return new QueryResult
            {
                Query = query,
                Answer = answer,
                VectorResults = vectorResults.Take(10).ToList(),
                KnowledgeSubgraphs = rankedSubgraphs.Take(5).Select(x => x.sg).ToList(),
                LatencyMs = latency,
                Timestamp = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to execute query: {Query}", query);
            throw;
        }
    }

    private async Task<List<string>> ExtractQueryEntitiesAsync(
        string query, 
        CancellationToken cancellationToken)
    {
        // Use NER model to extract entities from query
        var extracted = await _nerModel.ExtractEntitiesAsync(query, cancellationToken);
        if (extracted.Any())
        {
            return extracted.Select(e => e.Text).ToList();
        }

        // Fallback to capitalized words if NER fails
        var words = query.Split(' ');
        return words.Where(w => char.IsUpper(w.FirstOrDefault())).ToList();
    }

    private async Task<List<KnowledgeSubgraph>> ExtractKnowledgeSubgraphsAsync(
        List<string> queryEntities,
        List<SearchResult> vectorResults,
        CancellationToken cancellationToken)
    {
        var subgraphs = new List<KnowledgeSubgraph>();

        // Strategy 1: Subgraphs from query entities
        foreach (var entityText in queryEntities.Take(5))
        {
            var subgraph = await FindSubgraphForEntityAsync(entityText, cancellationToken);
            if (subgraph != null)
            {
                subgraphs.Add(subgraph);
            }
        }

        return subgraphs.Take(_maxSubgraphs).ToList();
    }

    private async Task<KnowledgeSubgraph?> FindSubgraphForEntityAsync(
        string entityText,
        CancellationToken cancellationToken)
    {
        var startNode = await _graphStorage.GetNodeByTextAsync(entityText, cancellationToken);
        if (startNode == null) return null;

        return await BuildSubgraphFromNodeAsync(startNode, cancellationToken);
    }

    private async Task<KnowledgeSubgraph?> BuildSubgraphFromNodeAsync(
        GraphNode startNode,
        CancellationToken cancellationToken)
    {
        try
        {
            // Traverse from start node
            var allNodes = await _graphStorage.TraverseAsync(startNode.Id, 2, cancellationToken);
            
            if (allNodes.Count == 0)
            {
                return null;
            }

            // Create subgraph from nodes
            var entities = new List<Entity>();
            var relations = new List<Relation>();

            foreach (var node in allNodes.Take(20)) // Limit subgraph size
            {
                var nodeDataJson = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(node.Data);
                if (nodeDataJson != null)
                {
                    var entity = new Entity
                    {
                        Id = node.Id,
                        TenantId = node.TenantId,
                        IndexName = node.IndexName,
                        Text = nodeDataJson.GetValueOrDefault("Text").GetString() ?? "",
                        Type = nodeDataJson.GetValueOrDefault("Type").GetString() ?? "ENTITY",
                        SourceChunkId = nodeDataJson.GetValueOrDefault("SourceChunkId").GetString() ?? ""
                    };
                    entities.Add(entity);
                }

                // Get edges
                var edges = await _graphStorage.GetOutgoingEdgesAsync(node.Id, cancellationToken);
                foreach (var edge in edges)
                {
                    var edgeDataJson = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(edge.Data);
                    if (edgeDataJson != null)
                    {
                        var relation = new Relation
                        {
                            Id = Guid.NewGuid().ToString(),
                            TenantId = node.TenantId,
                            IndexName = node.IndexName,
                            SourceEntityId = edge.SourceId,
                            TargetEntityId = edge.TargetId,
                            RelationType = edgeDataJson.GetValueOrDefault("RelationType").GetString() ?? "RELATED_TO",
                            Confidence = edgeDataJson.GetValueOrDefault("Confidence").GetSingle(),
                            SourceChunkId = edgeDataJson.GetValueOrDefault("SourceChunkId").GetString() ?? ""
                        };
                        relations.Add(relation);
                    }
                }
            }

            if (entities.Count == 0)
            {
                return null;
            }

            return new KnowledgeSubgraph
            {
                Id = Guid.NewGuid().ToString(),
                Entities = entities,
                Relations = relations
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to build subgraph from node {NodeId}", startNode.Id);
            return null;
        }
    }

    private List<CombinedResult> CombineResults(
        List<SearchResult> vectorResults,
        List<(KnowledgeSubgraph Subgraph, float Score)> rankedSubgraphs)
    {
        var combined = new List<CombinedResult>();

        // Combine vector results with graph results
        var maxVectorScore = vectorResults.Any() ? vectorResults.Max(r => r.Score) : 1.0f;
        var maxGraphScore = rankedSubgraphs.Any() ? rankedSubgraphs.Max(r => r.Score) : 1.0f;
        if (maxVectorScore == 0) maxVectorScore = 1.0f;
        if (maxGraphScore == 0) maxGraphScore = 1.0f;

        foreach (var vectorResult in vectorResults)
        {
            var normalizedVectorScore = vectorResult.Score / maxVectorScore;
            var finalScore = normalizedVectorScore * _vectorSearchWeight;

            combined.Add(new CombinedResult
            {
                ChunkId = vectorResult.Id,
                VectorScore = vectorResult.Score,
                GraphScore = 0,
                CombinedScore = finalScore
            });
        }

        foreach (var (subgraph, score) in rankedSubgraphs)
        {
            var normalizedGraphScore = score / maxGraphScore;
            var finalScore = normalizedGraphScore * _graphSearchWeight;

            // Try to match with existing vector results
            var chunkIds = subgraph.Entities.Select(e => e.SourceChunkId).Distinct();
            foreach (var chunkId in chunkIds)
            {
                if (string.IsNullOrEmpty(chunkId)) continue;

                var existing = combined.FirstOrDefault(c => c.ChunkId == chunkId);
                if (existing != null)
                {
                    existing.GraphScore = Math.Max(existing.GraphScore, score);
                    existing.CombinedScore += finalScore;
                }
                else
                {
                    combined.Add(new CombinedResult
                    {
                        ChunkId = chunkId,
                        VectorScore = 0,
                        GraphScore = score,
                        CombinedScore = finalScore
                    });
                }
            }
        }

        return combined.OrderByDescending(c => c.CombinedScore).ToList();
    }

    private async Task<string> GenerateAnswerAsync(
        string query,
        List<CombinedResult> results,
        CancellationToken cancellationToken)
    {
        if (results.Count == 0)
        {
            return "No relevant information found.";
        }

        // Construct prompt
        var sb = new StringBuilder();
        sb.AppendLine("Answer the question based on the following context:");
        sb.AppendLine();

        foreach (var result in results.Take(5))
        {
            // In a real app, we would fetch the chunk content here
            // Assuming result has some metadata or we fetch it
            sb.AppendLine($"Context (Score: {result.CombinedScore:F2}): [Chunk {result.ChunkId}]"); 
        }
        
        sb.AppendLine();
        sb.AppendLine($"Question: {query}");
        sb.AppendLine("Answer:");

        var prompt = sb.ToString();

        // Call LLM
        return await _llmGeneratorModel.GenerateAnswerAsync(prompt, _maxGenerationTokens, cancellationToken);
    }
}
