using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// ONNX model wrapper for Relation Extraction
/// Uses entity-centric dependency tree approach
/// </summary>
public sealed class RelationExtractionModel : OnnxModelBase, IRelationExtractionModel
{
    private readonly UncasedTokenizer _tokenizer;
    
    public RelationExtractionModel(
        string modelPath,
        bool useGpu,
        int gpuDeviceId,
        int maxPoolSize,
        UncasedTokenizer tokenizer,
        ILogger<RelationExtractionModel> logger)
        : base(modelPath, useGpu, gpuDeviceId, maxPoolSize, logger)
    {
        // Initialize BERT tokenizer
        _tokenizer = tokenizer;
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
                for (int j = 0; j < entities.Count; j++)
                {
                    if (i == j) continue; // Skip self-relations

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
        // 1. Insert entity markers
        string markedText = InsertEntityMarkers(text, entity1, entity2);

        // 2. Prepare input tensors
        var (inputIds, attentionMask, tokenTypeIds) = PrepareInputTensors(markedText);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };

        if (session.InputMetadata.ContainsKey("token_type_ids"))
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds));
        }

        using var results = session.Run(inputs);
        
        // 3. Process outputs
        // The model outputs "logits" (batch_size, num_labels)
        // Prefer fetching by name if possible, otherwise take first
        var logitsTensor = results.FirstOrDefault(r => r.Name == "logits") ?? results.First();
        var logits = logitsTensor.AsEnumerable<float>().ToArray();
        
        // Calculate softmax to get probabilities
        var probabilities = Softmax(logits);
        
        // Get the predicted relation type (index with max probability)
        int predictedTypeIdx = Array.IndexOf(probabilities, probabilities.Max());
        string relationType = GetRelationTypeName(predictedTypeIdx);
        float confidence = probabilities[predictedTypeIdx];

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

    private string InsertEntityMarkers(string text, ExtractedEntity ent1, ExtractedEntity ent2)
    {
        var markers = new List<(int Index, string Marker, bool IsStart, int EntityStart, int EntityEnd)>();
        
        markers.Add((ent1.StartPosition, "[E1]", true, ent1.StartPosition, ent1.EndPosition));
        markers.Add((ent1.EndPosition, "[/E1]", false, ent1.StartPosition, ent1.EndPosition));
        
        markers.Add((ent2.StartPosition, "[E2]", true, ent2.StartPosition, ent2.EndPosition));
        markers.Add((ent2.EndPosition, "[/E2]", false, ent2.StartPosition, ent2.EndPosition));

        markers.Sort((a, b) => 
        {
            // 1. Index Descending
            int cmp = b.Index.CompareTo(a.Index);
            if (cmp != 0) return cmp;

            // 2. Tie-breaking at same index
            if (a.IsStart != b.IsStart)
            {
                // Process Start before End to get [End][Start] sequence
                return b.IsStart.CompareTo(a.IsStart); 
            }

            if (a.IsStart) // Both are Start
            {
                // Start match: We want [Outer][Inner].
                // Insert [Inner] then [Outer].
                // Inner has Smaller End (if Starts match).
                // So process Inner (Smaller End) first.
                return a.EntityEnd.CompareTo(b.EntityEnd);
            }
            else // Both are End
            {
                // End match: We want [/Inner][/Outer].
                // Insert [/Outer] then [/Inner].
                // Outer has Smaller Start (if Ends match).
                // So process Outer (Smaller Start) first.
                return a.EntityStart.CompareTo(b.EntityStart);
            }
        });

        var sb = new StringBuilder(text);
        foreach (var item in markers)
        {
            int safeIndex = Math.Clamp(item.Index, 0, sb.Length);
            sb.Insert(safeIndex, $" {item.Marker} ");
        }

        return sb.ToString();
    }

    private (Tensor<long> InputIds, Tensor<long> AttentionMask, Tensor<long> TokenTypeIds) 
        PrepareInputTensors(string text)
    {
        const int maxLength = 512;

        // Tokenize the text
        var tokens = _tokenizer.Tokenize(text);
        var encoded = _tokenizer.Encode(tokens.Count() + 2, text); // +2 for [CLS] and [SEP]
        
        var inputIds = new long[1][];
        var attentionMask = new long[1][];
        var tokenTypeIds = new long[1][];

        inputIds[0] = new long[maxLength];
        attentionMask[0] = new long[maxLength];
        tokenTypeIds[0] = new long[maxLength];
        
        // Copy encoded tokens
        var lengthToCopy = Math.Min(encoded.Count, maxLength);
        for (int j = 0; j < lengthToCopy; j++)
        {
            inputIds[0][j] = encoded[j].InputIds;
            attentionMask[0][j] = encoded[j].AttentionMask;
            tokenTypeIds[0][j] = encoded[j].TokenTypeIds;
        }

        return (
            CreateTensor(inputIds, new[] { 1, maxLength }),
            CreateTensor(attentionMask, new[] { 1, maxLength }),
            CreateTensor(tokenTypeIds, new[] { 1, maxLength })
        );
    }

    private float[] Softmax(float[] logits)
    {
        if (logits.Length == 0) return Array.Empty<float>();
        var maxLogit = logits.Max();
        var exp = logits.Select(x => (float)Math.Exp(x - maxLogit)).ToArray();
        var sumExp = exp.Sum();
        return exp.Select(x => x / sumExp).ToArray();
    }

    private string GetRelationTypeName(int index)
    {
        // Common relation types from TACRED dataset
        // This should match the training labels.json
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

