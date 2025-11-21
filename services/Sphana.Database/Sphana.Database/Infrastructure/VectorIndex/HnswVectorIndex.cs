using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sphana.Database.Infrastructure.VectorIndex;

/// <summary>
/// In-memory HNSW (Hierarchical Navigable Small World) vector index implementation
/// Optimized for low-latency similarity search
/// </summary>
public sealed class HnswVectorIndex : IVectorIndex
{
    private readonly int _dimension;
    private readonly int _m; // Number of bi-directional links
    private readonly int _efConstruction; // Size of dynamic candidate list during construction
    private readonly int _efSearch; // Size of dynamic candidate list during search
    private readonly DistanceMetric _distanceMetric;
    private readonly bool _normalize;
    private readonly ILogger<HnswVectorIndex> _logger;

    private readonly ConcurrentDictionary<string, HnswNode> _nodes;
    private readonly SemaphoreSlim _indexLock;
    private HnswNode? _entryPoint;
    private int _maxLevel;

    public int Count => _nodes.Count;

    public HnswVectorIndex(
        int dimension,
        int m = 16,
        int efConstruction = 200,
        int efSearch = 50,
        DistanceMetric distanceMetric = DistanceMetric.Cosine,
        bool normalize = true,
        ILogger<HnswVectorIndex> logger = null!)
    {
        _dimension = dimension;
        _m = m;
        _efConstruction = efConstruction;
        _efSearch = efSearch;
        _distanceMetric = distanceMetric;
        _normalize = normalize;
        _logger = logger;

        _nodes = new ConcurrentDictionary<string, HnswNode>();
        _indexLock = new SemaphoreSlim(1, 1);
        _maxLevel = 0;
    }

    public async Task AddAsync(string id, float[] vector, CancellationToken cancellationToken = default)
    {
        if (vector.Length != _dimension)
        {
            throw new ArgumentException($"Vector dimension mismatch. Expected {_dimension}, got {vector.Length}");
        }

        var normalizedVector = _normalize ? Normalize(vector) : vector;
        var level = GetRandomLevel();

        var node = new HnswNode(id, normalizedVector, level);

        await _indexLock.WaitAsync(cancellationToken);
        try
        {
            if (_nodes.TryAdd(id, node))
            {
                if (_entryPoint == null)
                {
                    _entryPoint = node;
                    _maxLevel = level;
                }
                else
                {
                    Insert(node);
                }
            }
        }
        finally
        {
            _indexLock.Release();
        }
    }

    public async Task AddBatchAsync(
        IEnumerable<(string Id, float[] Vector)> items, 
        CancellationToken cancellationToken = default)
    {
        foreach (var (id, vector) in items)
        {
            await AddAsync(id, vector, cancellationToken);
        }
    }

    public async Task<List<SearchResult>> SearchAsync(
        float[] queryVector, 
        int topK, 
        CancellationToken cancellationToken = default)
    {
        if (queryVector.Length != _dimension)
        {
            throw new ArgumentException($"Query vector dimension mismatch. Expected {_dimension}, got {queryVector.Length}");
        }

        if (_entryPoint == null || Count == 0)
        {
            return new List<SearchResult>();
        }

        var normalizedQuery = _normalize ? Normalize(queryVector) : queryVector;

        var results = SearchLayer(normalizedQuery, _entryPoint, _efSearch, 0);
        
        return results
            .OrderBy(x => x.Distance)
            .Take(topK)
            .Select(x => new SearchResult
            {
                Id = x.Node.Id,
                Score = 1.0f - x.Distance, // Convert distance to similarity score
                Vector = x.Node.Vector
            })
            .ToList();
    }

    public Task<bool> RemoveAsync(string id, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(_nodes.TryRemove(id, out _));
    }

    public async Task SaveAsync(string path, CancellationToken cancellationToken = default)
    {
        await _indexLock.WaitAsync(cancellationToken);
        try
        {
            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            await using var stream = File.Create(path);
            await using var writer = new BinaryWriter(stream);

            // Write header
            writer.Write(_dimension);
            writer.Write(_m);
            writer.Write(_maxLevel);
            writer.Write(_nodes.Count);

            // Write nodes
            foreach (var node in _nodes.Values)
            {
                writer.Write(node.Id);
                writer.Write(node.Level);
                
                foreach (var value in node.Vector)
                {
                    writer.Write(value);
                }

                // Write connections per level
                for (int level = 0; level <= node.Level; level++)
                {
                    var connections = node.Connections[level];
                    writer.Write(connections.Count);
                    foreach (var conn in connections)
                    {
                        writer.Write(conn.Id);
                    }
                }
            }

            _logger?.LogInformation("HNSW index saved to {Path} with {Count} nodes", path, _nodes.Count);
        }
        finally
        {
            _indexLock.Release();
        }
    }

    public async Task LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(path))
        {
            _logger?.LogWarning("HNSW index file not found: {Path}", path);
            return;
        }

        await _indexLock.WaitAsync(cancellationToken);
        try
        {
            _nodes.Clear();

            await using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);

            // Read header
            var dimension = reader.ReadInt32();
            var m = reader.ReadInt32();
            _maxLevel = reader.ReadInt32();
            var count = reader.ReadInt32();

            if (dimension != _dimension || m != _m)
            {
                throw new InvalidOperationException(
                    $"Index configuration mismatch. Expected (dim={_dimension}, m={_m}), got (dim={dimension}, m={m})");
            }

            // First pass: read nodes
            var nodeList = new List<HnswNode>();
            for (int i = 0; i < count; i++)
            {
                var id = reader.ReadString();
                var level = reader.ReadInt32();
                
                var vector = new float[_dimension];
                for (int j = 0; j < _dimension; j++)
                {
                    vector[j] = reader.ReadSingle();
                }

                var node = new HnswNode(id, vector, level);
                nodeList.Add(node);
                _nodes[id] = node;

                // Skip connections for now
                for (int l = 0; l <= level; l++)
                {
                    var connCount = reader.ReadInt32();
                    for (int c = 0; c < connCount; c++)
                    {
                        reader.ReadString(); // Skip connection ID
                    }
                }
            }

            // Second pass: restore connections
            stream.Position = 4 * sizeof(int); // Reset to after header
            for (int i = 0; i < count; i++)
            {
                var id = reader.ReadString();
                var level = reader.ReadInt32();
                
                // Skip vector
                stream.Position += _dimension * sizeof(float);

                var node = _nodes[id];
                for (int l = 0; l <= level; l++)
                {
                    var connCount = reader.ReadInt32();
                    for (int c = 0; c < connCount; c++)
                    {
                        var connId = reader.ReadString();
                        if (_nodes.TryGetValue(connId, out var connNode))
                        {
                            node.Connections[l].Add(connNode);
                        }
                    }
                }
            }

            _entryPoint = nodeList.FirstOrDefault();
            _logger?.LogInformation("HNSW index loaded from {Path} with {Count} nodes", path, _nodes.Count);
        }
        finally
        {
            _indexLock.Release();
        }
    }

    private void Insert(HnswNode node)
    {
        var nearestNeighbors = new List<(HnswNode Node, float Distance)>();
        
        // Search for nearest neighbors at all levels
        var currentNearest = _entryPoint;
        for (int level = _maxLevel; level > node.Level; level--)
        {
            var candidates = SearchLayer(node.Vector, currentNearest!, 1, level);
            if (candidates.Count > 0)
            {
                currentNearest = candidates[0].Node;
            }
        }

        for (int level = node.Level; level >= 0; level--)
        {
            var candidates = SearchLayer(node.Vector, currentNearest!, _efConstruction, level);
            var mConnections = level == 0 ? _m * 2 : _m;

            // Select M nearest neighbors
            var neighbors = candidates
                .OrderBy(x => x.Distance)
                .Take(mConnections)
                .ToList();

            // Add bidirectional links
            foreach (var (neighbor, _) in neighbors)
            {
                node.Connections[level].Add(neighbor);
                
                // Only add reciprocal link if neighbor has this level
                if (neighbor.Level >= level)
                {
                    neighbor.Connections[level].Add(node);

                    // Prune connections if needed
                    if (neighbor.Connections[level].Count > mConnections)
                    {
                        PruneConnections(neighbor, level, mConnections);
                    }
                }
            }

            if (candidates.Count > 0)
            {
                currentNearest = candidates[0].Node;
            }
        }

        if (node.Level > _maxLevel)
        {
            _maxLevel = node.Level;
            _entryPoint = node;
        }
    }

    private List<(HnswNode Node, float Distance)> SearchLayer(
        float[] query, 
        HnswNode entryPoint, 
        int ef, 
        int level)
    {
        var visited = new HashSet<string>();
        var candidates = new PriorityQueue<HnswNode, float>();
        var results = new PriorityQueue<HnswNode, float>(Comparer<float>.Create((a, b) => b.CompareTo(a))); // Max heap

        var entryDist = ComputeDistance(query, entryPoint.Vector);
        candidates.Enqueue(entryPoint, entryDist);
        results.Enqueue(entryPoint, entryDist);
        visited.Add(entryPoint.Id);

        while (candidates.Count > 0)
        {
            var current = candidates.Dequeue();
            var currentDist = ComputeDistance(query, current.Vector);

            if (currentDist > ComputeDistance(query, results.Peek().Vector))
            {
                break;
            }

            foreach (var neighbor in current.Level >= level ? current.Connections[level] : Enumerable.Empty<HnswNode>())
            {
                if (visited.Add(neighbor.Id))
                {
                    var neighborDist = ComputeDistance(query, neighbor.Vector);
                    
                    if (results.Count < ef || neighborDist < ComputeDistance(query, results.Peek().Vector))
                    {
                        candidates.Enqueue(neighbor, neighborDist);
                        results.Enqueue(neighbor, neighborDist);

                        if (results.Count > ef)
                        {
                            results.Dequeue();
                        }
                    }
                }
            }
        }

        var resultList = new List<(HnswNode Node, float Distance)>();
        while (results.Count > 0)
        {
            var node = results.Dequeue();
            resultList.Add((node, ComputeDistance(query, node.Vector)));
        }

        return resultList;
    }

    private void PruneConnections(HnswNode node, int level, int maxConnections)
    {
        if (node.Connections[level].Count <= maxConnections)
        {
            return;
        }

        var distances = node.Connections[level]
            .Select(n => (Node: n, Distance: ComputeDistance(node.Vector, n.Vector)))
            .OrderBy(x => x.Distance)
            .Take(maxConnections)
            .ToList();

        node.Connections[level].Clear();
        foreach (var (neighbor, _) in distances)
        {
            node.Connections[level].Add(neighbor);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float ComputeDistance(float[] a, float[] b)
    {
        return _distanceMetric switch
        {
            DistanceMetric.Cosine => 1.0f - DotProduct(a, b), // Assuming normalized vectors
            DistanceMetric.Euclidean => EuclideanDistance(a, b),
            DistanceMetric.DotProduct => -DotProduct(a, b), // Negate for min heap
            _ => throw new InvalidOperationException($"Unsupported distance metric: {_distanceMetric}")
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotProduct(float[] a, float[] b)
    {
        float result = 0;
        for (int i = 0; i < a.Length; i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float EuclideanDistance(float[] a, float[] b)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float)Math.Sqrt(sum);
    }

    private static float[] Normalize(float[] vector)
    {
        var magnitude = Math.Sqrt(vector.Sum(x => x * x));
        if (magnitude == 0) return vector;

        var normalized = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = (float)(vector[i] / magnitude);
        }
        return normalized;
    }

    private int GetRandomLevel()
    {
        var ml = 1.0 / Math.Log(_m);
        return (int)(-Math.Log(Random.Shared.NextDouble()) * ml);
    }

    private sealed class HnswNode
    {
        public string Id { get; }
        public float[] Vector { get; }
        public int Level { get; }
        public List<HashSet<HnswNode>> Connections { get; }

        public HnswNode(string id, float[] vector, int level)
        {
            Id = id;
            Vector = vector;
            Level = level;
            Connections = new List<HashSet<HnswNode>>();
            
            for (int i = 0; i <= level; i++)
            {
                Connections.Add(new HashSet<HnswNode>());
            }
        }
    }
}

public enum DistanceMetric
{
    Cosine,
    Euclidean,
    DotProduct
}

