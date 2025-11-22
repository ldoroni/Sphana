using Sphana.Database.Models;
using Sphana.Database.Models.KnowledgeGraph;
using Sphana.Database.Infrastructure.Onnx;
using Sphana.Database.Infrastructure.VectorIndex;
using Sphana.Database.Infrastructure.GraphStorage;
using System.Text.Json;

namespace Sphana.Database.Services;

/// <summary>
/// Service for ingesting documents into the NRDB system
/// Handles chunking, embedding generation, relation extraction, and indexing
/// </summary>
public sealed class DocumentIngestionService : IDocumentIngestionService
{
    private readonly IEmbeddingModel _embeddingModel;
    private readonly IRelationExtractionModel _relationExtractionModel;
    private readonly INerModel _nerModel;
    private readonly IVectorIndex _vectorIndex;
    private readonly IGraphStorage _graphStorage;
    private readonly ILogger<DocumentIngestionService> _logger;
    private readonly int _chunkSize;
    private readonly int _chunkOverlap;
    private readonly float _minRelationConfidence;

    public DocumentIngestionService(
        IEmbeddingModel embeddingModel,
        IRelationExtractionModel relationExtractionModel,
        INerModel nerModel,
        IVectorIndex vectorIndex,
        IGraphStorage graphStorage,
        ILogger<DocumentIngestionService> logger,
        int chunkSize = 512,
        int chunkOverlap = 50,
        float minRelationConfidence = 0.5f)
    {
        _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
        _relationExtractionModel = relationExtractionModel ?? throw new ArgumentNullException(nameof(relationExtractionModel));
        _nerModel = nerModel ?? throw new ArgumentNullException(nameof(nerModel));
        _vectorIndex = vectorIndex ?? throw new ArgumentNullException(nameof(vectorIndex));
        _graphStorage = graphStorage ?? throw new ArgumentNullException(nameof(graphStorage));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _chunkSize = chunkSize;
        _chunkOverlap = chunkOverlap;
        _minRelationConfidence = minRelationConfidence;
    }

    /// <summary>
    /// Ingest a single document
    /// </summary>
    public async Task<string> IngestDocumentAsync(
        Document document, 
        CancellationToken cancellationToken = default)
    {
        // Validate input
        if (document == null)
        {
            throw new ArgumentNullException(nameof(document));
        }

        _logger.LogInformation("Starting ingestion of document {DocumentId} - {Title}", 
            document.Id, document.Title);

        try
        {
            // Step 1: Chunk the document
            var chunks = await ChunkDocumentAsync(document, cancellationToken);
            _logger.LogDebug("Created {ChunkCount} chunks from document {DocumentId}", 
                chunks.Count, document.Id);

            // Step 2: Generate embeddings for all chunks
            var chunkTexts = chunks.Select(c => c.Content).ToArray();
            var embeddings = await _embeddingModel.GenerateEmbeddingsAsync(chunkTexts, cancellationToken);

            for (int i = 0; i < chunks.Count; i++)
            {
                chunks[i].Embedding = embeddings[i];
                chunks[i].QuantizedEmbedding = EmbeddingModel.Quantize(embeddings[i]);
            }

            _logger.LogDebug("Generated embeddings for {ChunkCount} chunks", chunks.Count);

            // Step 3: Add chunks to vector index
            var vectorTasks = chunks.Select(chunk => 
                _vectorIndex.AddAsync(chunk.Id, chunk.Embedding!, cancellationToken));
            await Task.WhenAll(vectorTasks);

            _logger.LogDebug("Added {ChunkCount} chunks to vector index", chunks.Count);

            // Step 4: Extract entities and relations from each chunk
            var allEntities = new List<Entity>();
            var allRelations = new List<Relation>();

            foreach (var chunk in chunks)
            {
                var (entities, relations) = await ExtractKnowledgeAsync(chunk, cancellationToken);
                allEntities.AddRange(entities);
                allRelations.AddRange(relations);
            }

            _logger.LogDebug("Extracted {EntityCount} entities and {RelationCount} relations",
                allEntities.Count, allRelations.Count);

            // Step 5: Add entities and relations to knowledge graph
            await BuildKnowledgeGraphAsync(allEntities, allRelations, cancellationToken);

            _logger.LogInformation("Successfully ingested document {DocumentId}", document.Id);

            return document.Id;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to ingest document {DocumentId}", document.Id);
            throw;
        }
    }

    /// <summary>
    /// Ingest multiple documents in batch
    /// </summary>
    public async Task<List<string>> IngestDocumentsBatchAsync(
        IEnumerable<Document> documents, 
        int maxConcurrency = 10,
        CancellationToken cancellationToken = default)
    {
        var semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency);
        var tasks = documents.Select(async doc =>
        {
            await semaphore.WaitAsync(cancellationToken);
            try
            {
                return await IngestDocumentAsync(doc, cancellationToken);
            }
            finally
            {
                semaphore.Release();
            }
        });

        var results = await Task.WhenAll(tasks);
        return results.ToList();
    }

    private async Task<List<DocumentChunk>> ChunkDocumentAsync(
        Document document, 
        CancellationToken cancellationToken)
    {
        var chunks = new List<DocumentChunk>();
        var content = document.Content;

        // Simple token-based chunking
        // In a real implementation, use a proper tokenizer
        var tokens = content.Split(new[] { ' ', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        
        for (int i = 0; i < tokens.Length; i += (_chunkSize - _chunkOverlap))
        {
            var chunkTokens = tokens.Skip(i).Take(_chunkSize).ToArray();
            var chunkText = string.Join(" ", chunkTokens);

            if (string.IsNullOrWhiteSpace(chunkText))
            {
                continue;
            }

            var chunk = new DocumentChunk
            {
                Id = $"{document.Id}_{i / (_chunkSize - _chunkOverlap)}",
                DocumentId = document.Id,
                TenantId = document.TenantId,
                IndexName = document.IndexName,
                Content = chunkText,
                ChunkIndex = i / (_chunkSize - _chunkOverlap),
                StartPosition = i,
                EndPosition = Math.Min(i + _chunkSize, tokens.Length),
                EmbeddingDimension = 384 // Default dimension
            };

            chunks.Add(chunk);
        }

        return chunks;
    }

    private async Task<(List<Entity> Entities, List<Relation> Relations)> ExtractKnowledgeAsync(
        DocumentChunk chunk,
        CancellationToken cancellationToken)
    {
        // Use NER model to extract entities
        var extractedEntities = await _nerModel.ExtractEntitiesAsync(chunk.Content, cancellationToken);

        if (extractedEntities.Count < 2)
        {
            return (new List<Entity>(), new List<Relation>());
        }

        // Extract relations using the RE model
        var extractedRelations = await _relationExtractionModel.ExtractRelationsAsync(
            chunk.Content, 
            extractedEntities, 
            cancellationToken);

        // Filter by confidence threshold
        extractedRelations = extractedRelations
            .Where(r => r.Confidence >= _minRelationConfidence)
            .ToList();

        // Convert to domain models
        var entities = new List<Entity>();
        
        // Deduplicate entities by text for embedding generation to save compute
        var uniqueTexts = extractedEntities.Select(e => e.Text).Distinct().ToArray();
        var textEmbeddings = new Dictionary<string, float[]>();
        
        if (uniqueTexts.Length > 0)
        {
            var embeddings = await _embeddingModel.GenerateEmbeddingsAsync(uniqueTexts, cancellationToken);
            for (int i = 0; i < uniqueTexts.Length; i++)
            {
                textEmbeddings[uniqueTexts[i]] = embeddings[i];
            }
        }

        var entityMap = new Dictionary<string, Entity>(); // Text -> Entity

        foreach (var text in uniqueTexts)
        {
            // Find first type for this text
            var type = extractedEntities.FirstOrDefault(e => e.Text == text)?.Type ?? "ENTITY";
            
            var entity = new Entity
            {
                Id = Guid.NewGuid().ToString(), // Ideally hash the text/type
                TenantId = chunk.TenantId,
                IndexName = chunk.IndexName,
                Text = text,
                Type = type, // Assigned in initializer
                SourceChunkId = chunk.Id,
                Embedding = textEmbeddings[text],
                QuantizedEmbedding = EmbeddingModel.Quantize(textEmbeddings[text])
            };

            entities.Add(entity);
            entityMap[text] = entity;
        }

        var relations = extractedRelations.Select(r => new Relation
        {
            Id = Guid.NewGuid().ToString(),
            TenantId = chunk.TenantId,
            IndexName = chunk.IndexName,
            SourceEntityId = entityMap[r.SourceEntity.Text].Id,
            TargetEntityId = entityMap[r.TargetEntity.Text].Id,
            RelationType = r.RelationType,
            Confidence = r.Confidence,
            SourceChunkId = chunk.Id
        }).ToList();

        return (entities, relations);
    }

    private async Task BuildKnowledgeGraphAsync(
        List<Entity> entities,
        List<Relation> relations,
        CancellationToken cancellationToken)
    {
        // Add entities as nodes
        var nodeIdMap = new Dictionary<string, string>();
        foreach (var entity in entities)
        {
            // Check if we already added this entity in this batch
            if (nodeIdMap.ContainsKey(entity.Id)) continue;

            var nodeData = JsonSerializer.Serialize(new
            {
                entity.Text,
                entity.Type,
                entity.SourceChunkId,
                entity.Properties
            });

            var nodeId = await _graphStorage.AddNodeAsync(
                entity.TenantId,
                entity.IndexName,
                nodeData,
                cancellationToken);

            nodeIdMap[entity.Id] = nodeId;
        }

        // Add relations as edges
        foreach (var relation in relations)
        {
            if (nodeIdMap.TryGetValue(relation.SourceEntityId, out var sourceNodeId) &&
                nodeIdMap.TryGetValue(relation.TargetEntityId, out var targetNodeId))
            {
                var edgeData = JsonSerializer.Serialize(new
                {
                    relation.RelationType,
                    relation.Confidence,
                    relation.SourceChunkId
                });

                await _graphStorage.AddEdgeAsync(sourceNodeId, targetNodeId, edgeData, cancellationToken);
            }
        }
    }
}
