using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// ONNX model wrapper for Named Entity Recognition (NER)
/// Uses BERT-based Token Classification model
/// </summary>
public sealed class NerModel : OnnxModelBase, INerModel
{
    private readonly UncasedTokenizer _tokenizer;
    private readonly string[] _labels;

    public NerModel(
        string modelPath,
        bool useGpu,
        int gpuDeviceId,
        int maxPoolSize,
        UncasedTokenizer tokenizer,
        ILogger<NerModel> logger)
        : base(modelPath, useGpu, gpuDeviceId, maxPoolSize, logger)
    {
        _tokenizer = tokenizer;
        // Standard CoNLL-2003 labels
        _labels = new[] { "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC" };
    }

    public async Task<List<ExtractedEntity>> ExtractEntitiesAsync(
        string text, 
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return new List<ExtractedEntity>();
        }

        var session = await AcquireSessionAsync(cancellationToken);
        try
        {
            // 1. Tokenize and Encode
            var (inputIds, attentionMask, tokenTypeIds, tokens) = PrepareInput(text);

            // 2. Run Inference
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
            var output = results.First().AsEnumerable<float>().ToArray();

            // 3. Decode Labels
            // Output shape: [1, seq_len, num_labels]
            int seqLen = inputIds.Dimensions[1];
            int numLabels = _labels.Length;
            
            // Assuming model output matches standard shape (batch, seq, labels)
            // If output is flattened, we need to stride.
            
            var predictedLabels = new List<string>();
            for (int i = 0; i < seqLen; i++)
            {
                // Find max logit for this token
                int maxIdx = 0;
                float maxVal = float.MinValue;
                for (int j = 0; j < numLabels; j++)
                {
                    // Verify index bounds
                    int idx = i * numLabels + j;
                    if (idx < output.Length)
                    {
                        if (output[idx] > maxVal)
                        {
                            maxVal = output[idx];
                            maxIdx = j;
                        }
                    }
                }
                predictedLabels.Add(_labels[maxIdx]);
            }

            // 4. Aggregate Entities
            return ReconstructEntities(text, tokens, predictedLabels);
        }
        finally
        {
            ReleaseSession(session);
        }
    }

    private (Tensor<long> InputIds, Tensor<long> AttentionMask, Tensor<long> TokenTypeIds, List<(string Token, int Offset)> Tokens) 
        PrepareInput(string text)
    {
        // Tokenize with offsets
        // BERTTokenizers doesn't expose offsets directly in Encode, so we do a manual pass
        // This is a simplification; proper BERT tokenization with offsets is complex.
        // We'll try to align tokens to text.
        
        var rawTokens = _tokenizer.Tokenize(text);
        var tokens = new List<(string Token, int Offset)>();
        
        int currentPos = 0;
        foreach (var tokenTuple in rawTokens)
        {
            // Token is tuple (Token, VocabularyIndex, SegmentIndex)
            var token = tokenTuple.Token;

            // Remove ## for subwords to find in text
            var cleanToken = token.StartsWith("##") ? token.Substring(2) : token;
            
            // Simple search from currentPos
            // Note: This might fail if tokenization normalization changed characters (e.g. accent removal)
            // For strict correctness we need a tokenizer that returns offsets.
            int pos = text.IndexOf(cleanToken, currentPos, StringComparison.OrdinalIgnoreCase);
            if (pos >= 0)
            {
                tokens.Add((token, pos));
                currentPos = pos + cleanToken.Length;
            }
            else
            {
                // Fallback: keep current pos (might involve skipped chars like spaces)
                tokens.Add((token, currentPos));
            }
        }

        const int maxLength = 512;
        var encoded = _tokenizer.Encode(tokens.Count + 2, text);

        // Flattened arrays for DenseTensor
        var inputIds = new long[maxLength];
        var attentionMask = new long[maxLength];
        var tokenTypeIds = new long[maxLength];

        var length = Math.Min(encoded.Count, maxLength);
        for (int i = 0; i < length; i++)
        {
            inputIds[i] = encoded[i].InputIds;
            attentionMask[i] = encoded[i].AttentionMask;
            tokenTypeIds[i] = encoded[i].TokenTypeIds;
        }

        return (
            new DenseTensor<long>(inputIds, new[] { 1, maxLength }),
            new DenseTensor<long>(attentionMask, new[] { 1, maxLength }),
            new DenseTensor<long>(tokenTypeIds, new[] { 1, maxLength }),
            tokens
        );
    }

    private List<ExtractedEntity> ReconstructEntities(
        string text, 
        List<(string Token, int Offset)> tokens, 
        List<string> labels)
    {
        var entities = new List<ExtractedEntity>();
        ExtractedEntity? currentEntity = null;
        string currentType = "";

        // Labels start from [CLS] at index 0, tokens align with labels[1..]
        // Truncate labels to match tokens length
        
        for (int i = 0; i < tokens.Count; i++)
        {
            // Label index is i + 1 because of [CLS]
            if (i + 1 >= labels.Count) break;
            
            string label = labels[i + 1];
            var (token, offset) = tokens[i];
            
            if (label == "O")
            {
                if (currentEntity != null)
                {
                    entities.Add(currentEntity);
                    currentEntity = null;
                }
                continue;
            }

            // B-TYPE or I-TYPE
            var parts = label.Split('-');
            if (parts.Length != 2) continue;
            
            string prefix = parts[0];
            string type = parts[1];

            if (prefix == "B")
            {
                if (currentEntity != null)
                {
                    entities.Add(currentEntity);
                }
                
                // Start new entity
                int end = offset + (token.StartsWith("##") ? token.Length - 2 : token.Length);
                // Try to capture full word from text if subword
                // Simplification: just use calculated offsets
                
                currentEntity = new ExtractedEntity
                {
                    Text = token.StartsWith("##") ? token.Substring(2) : token,
                    Type = type,
                    StartPosition = offset,
                    EndPosition = end
                };
                currentType = type;
            }
            else if (prefix == "I")
            {
                if (currentEntity != null && currentType == type)
                {
                    // Append to current
                    // Calculate length
                    int len = token.StartsWith("##") ? token.Length - 2 : token.Length;
                    
                    // Check for spacing in original text between previous end and current offset
                    // to correctly reconstruct Text
                    if (currentEntity.EndPosition < offset)
                    {
                        int gap = offset - currentEntity.EndPosition;
                        if (gap > 0)
                        {
                            string gapText = text.Substring(currentEntity.EndPosition, gap);
                            // Append gap text? NER usually doesn't span gaps unless they are spaces.
                            // But 'token' offsets found via search might jump.
                            // Let's just reconstruct from text using Start/End
                        }
                    }
                    
                    var newEnd = offset + len;
                    var fullText = text.Substring(currentEntity.StartPosition, newEnd - currentEntity.StartPosition);
                    
                    currentEntity = new ExtractedEntity 
                    { 
                        Text = fullText,
                        Type = type,
                        StartPosition = currentEntity.StartPosition,
                        EndPosition = newEnd
                    };
                }
                else
                {
                    // I- without B- (or type mismatch), start new? Or treat as B?
                    // Usually treat as B
                    int end = offset + (token.StartsWith("##") ? token.Length - 2 : token.Length);
                    currentEntity = new ExtractedEntity
                    {
                        Text = token.StartsWith("##") ? token.Substring(2) : token,
                        Type = type,
                        StartPosition = offset,
                        EndPosition = end
                    };
                    currentType = type;
                }
            }
        }

        if (currentEntity != null)
        {
            entities.Add(currentEntity);
        }

        return entities;
    }
}

