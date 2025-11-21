using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BERTTokenizers;
using Sphana.Database.Models.KnowledgeGraph;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// ONNX model wrapper for Relation Extraction
/// Uses entity-centric dependency tree approach
/// </summary>
public sealed class RelationExtractionModel : OnnxModelBase, IRelationExtractionModel
{
    private readonly BertUncasedBaseTokenizer _tokenizer;
    
    public RelationExtractionModel(
        string modelPath,
        bool useGpu,
        int gpuDeviceId,
        int maxPoolSize,
        ILogger<RelationExtractionModel> logger)
        : base(modelPath, useGpu, gpuDeviceId, maxPoolSize, logger)
    {
        // Initialize BERT tokenizer
        _tokenizer = new BertUncasedBaseTokenizer();
    }

    /// <summary>
    /// Extract relations from text
    /// </summary>
    public async Task<List<ExtractedRelation>> ExtractRelationsAsync(
        string text,
        List<ExtractedEntity> entities,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(text) || entities == null || entities.Count < 2)
        {
            return new List<ExtractedRelation>();
        }

        var session = await AcquireSessionAsync(cancellationToken);
        try
        {
            var relations = new List<ExtractedRelation>();

            // Consider all entity pairs
            for (int i = 0; i < entities.Count; i++)
            {
                for (int j = i + 1; j < entities.Count; j++)
                {
                    var relation = await ExtractRelationForPairAsync(
                        session, 
                        text, 
                        entities[i], 
                        entities[j], 
                        cancellationToken);

                    if (relation != null && relation.Confidence > 0.5f)
                    {
                        relations.Add(relation);
                    }
                }
            }

            return relations;
        }
        finally
        {
            ReleaseSession(session);
        }
    }

    private async Task<ExtractedRelation?> ExtractRelationForPairAsync(
        InferenceSession session,
        string text,
        ExtractedEntity entity1,
        ExtractedEntity entity2,
        CancellationToken cancellationToken)
    {
        // Prepare input tensors
        var (inputIds, attentionMask, entityPositions) = PrepareInputTensors(text, entity1, entity2);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
            NamedOnnxValue.CreateFromTensor("entity_positions", entityPositions)
        };

        using var results = session.Run(inputs);
        
        // Extract outputs: relation_type_logits [num_relation_types], confidence [1]
        var relationTypeLogits = results.First(r => r.Name == "relation_type_logits")
            .AsEnumerable<float>().ToArray();
        var confidence = results.First(r => r.Name == "confidence")
            .AsEnumerable<float>().First();

        // Get the predicted relation type (index with max logit)
        int predictedTypeIdx = Array.IndexOf(relationTypeLogits, relationTypeLogits.Max());
        string relationType = GetRelationTypeName(predictedTypeIdx);

        if (confidence < 0.5f || relationType == "NO_RELATION")
        {
            return null;
        }

        return new ExtractedRelation
        {
            SourceEntity = entity1,
            TargetEntity = entity2,
            RelationType = relationType,
            Confidence = confidence
        };
    }

    private (Tensor<long> InputIds, Tensor<long> AttentionMask, Tensor<long> EntityPositions) 
        PrepareInputTensors(string text, ExtractedEntity entity1, ExtractedEntity entity2)
    {
        const int maxLength = 512;

        // Tokenize the text
        var tokens = _tokenizer.Tokenize(text);
        var encoded = _tokenizer.Encode(tokens.Count() + 2, text); // +2 for [CLS] and [SEP]
        
        var inputIds = new long[1][];
        var attentionMask = new long[1][];
        var entityPositions = new long[1][];

        inputIds[0] = new long[maxLength];
        attentionMask[0] = new long[maxLength];
        
        // Copy encoded tokens
        var lengthToCopy = Math.Min(encoded.Count, maxLength);
        for (int j = 0; j < lengthToCopy; j++)
        {
            inputIds[0][j] = encoded[j].InputIds;
            attentionMask[0][j] = encoded[j].AttentionMask;
        }

        // Entity positions: [start1, end1, start2, end2]
        // Note: These positions need to be adjusted for tokenization
        // For simplicity, we're using character positions; in production, map to token positions
        entityPositions[0] = new long[4]
        {
            Math.Min(entity1.StartPosition, maxLength - 1),
            Math.Min(entity1.EndPosition, maxLength - 1),
            Math.Min(entity2.StartPosition, maxLength - 1),
            Math.Min(entity2.EndPosition, maxLength - 1)
        };

        return (
            CreateTensor(inputIds, new[] { 1, maxLength }),
            CreateTensor(attentionMask, new[] { 1, maxLength }),
            CreateTensor(entityPositions, new[] { 1, 4 })
        );
    }

    private string GetRelationTypeName(int index)
    {
        // Common relation types from TACRED dataset
        // This should be loaded from the model metadata
        var relationTypes = new[]
        {
            "NO_RELATION",
            "org:founded_by",
            "per:employee_of",
            "org:alternate_names",
            "per:cities_of_residence",
            "per:children",
            "per:title",
            "per:siblings",
            "per:religion",
            "per:age",
            "org:website",
            "per:stateorprovinces_of_residence",
            "org:member_of",
            "org:top_members/employees",
            "per:countries_of_residence",
            "org:city_of_headquarters",
            "org:members",
            "org:country_of_headquarters",
            "per:spouse",
            "org:stateorprovince_of_headquarters",
            "org:number_of_employees/members",
            "org:parents",
            "org:subsidiaries",
            "per:origin",
            "org:political/religious_affiliation",
            "per:other_family",
            "per:stateorprovince_of_birth",
            "org:dissolved",
            "per:date_of_death",
            "org:shareholders",
            "per:alternate_names",
            "per:parents",
            "per:schools_attended",
            "per:cause_of_death",
            "per:city_of_death",
            "per:stateorprovince_of_death",
            "org:founded",
            "per:country_of_birth",
            "per:date_of_birth",
            "per:city_of_birth",
            "per:charges"
        };

        return index >= 0 && index < relationTypes.Length 
            ? relationTypes[index] 
            : "NO_RELATION";
    }
}

/// <summary>
/// Represents an extracted entity from text
/// </summary>
public sealed class ExtractedEntity
{
    public required string Text { get; init; }
    public required string Type { get; init; }
    public required int StartPosition { get; init; }
    public required int EndPosition { get; init; }
}

/// <summary>
/// Represents an extracted relation between two entities
/// </summary>
public sealed class ExtractedRelation
{
    public required ExtractedEntity SourceEntity { get; init; }
    public required ExtractedEntity TargetEntity { get; init; }
    public required string RelationType { get; init; }
    public required float Confidence { get; init; }
}

