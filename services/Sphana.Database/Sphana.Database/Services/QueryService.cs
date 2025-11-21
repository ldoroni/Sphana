using Sphana.Database.Models.KnowledgeGraph;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Infrastructure.GraphStorage;
using System.Text.Json;

namespace Sphana.Database.Services;

/// <summary>
/// Service for querying documents using hybrid vector + graph retrieval
/// </summary>
public sealed class QueryService : IQueryService
{
    private readonly IEmbeddingModel _embeddingModel;
    private readonly IGnnRankerModel _gnnRankerModel;
    private readonly IVectorIndex _vectorIndex;
    private readonly IGraphStorage _graphStorage;
    private readonly ILogger<QueryService> _logger;
    private readonly float _vectorSearchWeight;
    private readonly float _graphSearchWeight;
    private readonly int _vectorSearchTopK;
    private readonly int _maxSubgraphs;

    public QueryService(
        IEmbeddingModel embeddingModel,
        IGnnRankerModel gnnRankerModel,
        IVectorIndex vectorIndex,
        IGraphStorage graphStorage,
        ILogger<QueryService> logger,
        float vectorSearchWeight = 0.6f,
        float graphSearchWeight = 0.4f,
        int vectorSearchTopK = 20,
        int maxSubgraphs = 10)
    {
        _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
        _gnnRankerModel = gnnRankerModel ?? throw new ArgumentNullException(nameof(gnnRankerModel));
        _vectorIndex = vectorIndex ?? throw new ArgumentNullException(nameof(vectorIndex));
        _graphStorage = graphStorage ?? throw new ArgumentNullException(nameof(graphStorage));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _vectorSearchWeight = vectorSearchWeight;
        _graphSearchWeight = graphSearchWeight;
        _vectorSearchTopK = vectorSearchTopK;
        _maxSubgraphs = maxSubgraphs;
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

            // Step 3: Extract entities from query (placeholder)
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

            // Step 7: Generate final answer (placeholder for LLM)
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
        // TODO: Implement proper entity extraction from query
        // For now, return capitalized words as entities
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

        // Strategy 2: Subgraphs from vector search results
        foreach (var result in vectorResults.Take(10))
        {
            // Extract chunk ID from result ID
            var chunkId = result.Id;
            var subgraph = await FindSubgraphForChunkAsync(chunkId, cancellationToken);
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
        // TODO: Implement proper entity lookup in the graph
        // For now, return null as placeholder
        return await Task.FromResult<KnowledgeSubgraph?>(null);
    }

    private async Task<KnowledgeSubgraph?> FindSubgraphForChunkAsync(
        string chunkId,
        CancellationToken cancellationToken)
    {
        // Find all nodes associated with this chunk
        // This is a simplified implementation
        try
        {
            // In a real implementation, maintain a chunk-to-nodes mapping
            // For now, traverse a portion of the graph
            var allNodes = await _graphStorage.TraverseAsync("", 2, cancellationToken);
            
            if (allNodes.Count == 0)
            {
                return null;
            }

            // Create subgraph from nodes
            var entities = new List<Entity>();
            var relations = new List<Relation>();

            foreach (var node in allNodes.Take(10))
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
            _logger.LogWarning(ex, "Failed to extract subgraph for chunk {ChunkId}", chunkId);
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
        // TODO: Implement LLM-based answer generation
        // For now, return a simple summary
        if (results.Count == 0)
        {
            return "No relevant information found.";
        }

        var topResults = results.Take(5);
        return $"Found {results.Count} relevant results. Top chunks: {string.Join(", ", topResults.Select(r => r.ChunkId))}";
    }
}

/// <summary>
/// Result of a query operation
/// </summary>
public sealed class QueryResult
{
    public required string Query { get; init; }
    public required string Answer { get; init; }
    public List<SearchResult> VectorResults { get; init; } = new();
    public List<KnowledgeSubgraph> KnowledgeSubgraphs { get; init; } = new();
    public double LatencyMs { get; init; }
    public DateTime Timestamp { get; init; }
}

/// <summary>
/// Combined result from vector and graph search
/// </summary>
public sealed class CombinedResult
{
    public required string ChunkId { get; init; }
    public float VectorScore { get; init; }
    public float GraphScore { get; set; }
    public float CombinedScore { get; set; }
}

