using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Models.KnowledgeGraph;

namespace Sphana.Database.Models;

public class QueryResult
{
    public required string Query { get; set; }
    public required string Answer { get; set; }
    public required List<SearchResult> VectorResults { get; set; }
    public required List<KnowledgeSubgraph> KnowledgeSubgraphs { get; set; }
    public double LatencyMs { get; set; }
    public DateTime Timestamp { get; set; }
}

