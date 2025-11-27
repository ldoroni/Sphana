using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using System.Text;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// ONNX model wrapper for LLM generation
/// </summary>
public sealed class LlmGeneratorModel : OnnxModelBase, ILlmGeneratorModel
{
    private readonly Microsoft.ML.Tokenizers.Tokenizer _tokenizer;
    private readonly int _bosTokenId;
    private readonly int _eosTokenId;
    private readonly bool _logTokenizedText;

    public LlmGeneratorModel(
        string modelPath,
        string tokenizerPath,
        bool useGpu,
        int gpuDeviceId,
        int maxPoolSize,
        bool logTokenizedText,
        ILogger<LlmGeneratorModel> logger)
        : base(modelPath, useGpu, gpuDeviceId, maxPoolSize, logger)
    {
        _logTokenizedText = logTokenizedText;
        
        // Load tokenizer - support both tokenizer.model (SentencePiece) and tokenizer.json (Llama 3.2)
        var tokenizerDir = Path.GetDirectoryName(tokenizerPath) ?? "";
        var tokenizerModelPath = Path.Combine(tokenizerDir, "tokenizer.model");
        var tokenizerJsonPath = tokenizerPath; // Should be tokenizer.json from config
        
        if (File.Exists(tokenizerModelPath))
        {
            // Load SentencePiece tokenizer.model (Gemma, older Llama models)
            using var stream = File.OpenRead(tokenizerModelPath);
            _tokenizer = LlamaTokenizer.Create(
                stream,
                addBeginOfSentence: true,
                addEndOfSentence: false
            );
            
            _bosTokenId = 128000; // Llama 3 BOS token (compatible with Gemma's 2)
            _eosTokenId = 128001; // Llama 3 EOS token (compatible with Gemma's 1)
            
            logger.LogInformation("Loaded SentencePiece tokenizer from {TokenizerPath}, BOS={BOS}, EOS={EOS}", 
                tokenizerModelPath, _bosTokenId, _eosTokenId);
        }
        else if (File.Exists(tokenizerJsonPath))
        {
            // Load from tokenizer.json (Llama 3.2, newer models)
            // Microsoft.ML.Tokenizers has limited support for arbitrary tokenizer.json
            // Using Tiktoken as an approximation for Llama 3.2
            logger.LogWarning("tokenizer.model not found, using Tiktoken approximation for Llama 3.2 (cl100k_base encoding)");
            
            _tokenizer = TiktokenTokenizer.CreateForModel("gpt-4"); // cl100k_base encoding, closest to Llama 3
            
            _bosTokenId = 128000; // Llama 3.2 BOS token
            _eosTokenId = 128001; // Llama 3.2 EOS token  
            
            logger.LogInformation("Loaded Tiktoken tokenizer (approximation), BOS={BOS}, EOS={EOS}", 
                _bosTokenId, _eosTokenId);
        }
        else
        {
            throw new FileNotFoundException(
                $"No compatible tokenizer found. Looked for:\n" +
                $"  - {tokenizerModelPath} (SentencePiece)\n" +
                $"  - {tokenizerJsonPath} (HuggingFace format)"
            );
        }
    }
    
    private List<int> Tokenize(string text)
    {
        var result = _tokenizer.EncodeToIds(text);
        return result.ToList();
    }
    
    private string DecodeTokens(List<int> tokenIds)
    {
        return _tokenizer.Decode(tokenIds) ?? string.Empty;
    }

    public async Task<string> GenerateAnswerAsync(
        string prompt, 
        int maxTokens, 
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return string.Empty;
        }

        var session = await AcquireSessionAsync(cancellationToken);
        try
        {
            // Get model metadata to find number of layers and KV cache dimensions
            var inputMetadata = session.InputMetadata;
            int numLayers = inputMetadata.Keys.Count(k => k.StartsWith("past_key_values.") && k.EndsWith(".key"));
            
            // Detect KV cache dimensions from model metadata
            // Shape is [batch_size, num_heads_kv, past_seq_len, head_dim]
            var firstKVKey = $"past_key_values.0.key";
            int numHeadsKV = 1;
            int headDim = 256;
            
            if (inputMetadata.ContainsKey(firstKVKey))
            {
                var kvShape = inputMetadata[firstKVKey].Dimensions;
                if (kvShape.Length >= 4)
                {
                    numHeadsKV = kvShape[1]; // Number of key/value heads
                    headDim = kvShape[3];    // Head dimension
                    _logger.LogInformation("Detected KV cache dimensions: num_heads_kv={NumHeads}, head_dim={HeadDim}", 
                        numHeadsKV, headDim);
                }
            }
            
            _logger.LogInformation("Model has {NumLayers} transformer layers", numLayers);

            // Tokenize prompt using Microsoft.ML.Tokenizers
            var currentInputIds = Tokenize(prompt).Select(id => (long)id).ToList();
            var currentMask = Enumerable.Repeat(1L, currentInputIds.Count).ToList();
            
            _logger.LogInformation("Tokenized prompt into {TokenCount} tokens: [{Tokens}]", 
                currentInputIds.Count, string.Join(", ", currentInputIds));

            // Debug logging: show decoded text
            if (_logTokenizedText)
            {
                var decodedText = DecodeTokens(currentInputIds.Select(id => (int)id).ToList());
                _logger.LogInformation("Decoded tokenized text: {DecodedText}", decodedText);
            }

            // Initialize KV cache for first pass
            Dictionary<string, Tensor<float>> pastKVCache = null;
            int totalSeqLen = 0;
            
            // Simple Greedy Generation Loop
            var generatedTokenIds = new List<int>();
            
            for (int step = 0; step < maxTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested) break;

                // For first iteration, use full prompt; after that, only last generated token
                int seqLen = (step == 0) ? currentInputIds.Count : 1;
                var inputIdsForStep = (step == 0) ? currentInputIds : new List<long> { generatedTokenIds.Last() };
                
                totalSeqLen = currentMask.Count;
                
                var inputIdsTensor = new DenseTensor<long>(new[] { 1, seqLen });
                var maskTensor = new DenseTensor<long>(new[] { 1, totalSeqLen });
                var positionIdsTensor = new DenseTensor<long>(new[] { 1, seqLen });
                
                for (int i = 0; i < seqLen; i++)
                {
                    inputIdsTensor[0, i] = inputIdsForStep[i];
                    positionIdsTensor[0, i] = totalSeqLen - seqLen + i;
                }
                
                for (int i = 0; i < totalSeqLen; i++)
                {
                    maskTensor[0, i] = currentMask[i];
                }

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                    NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor),
                    NamedOnnxValue.CreateFromTensor("position_ids", positionIdsTensor)
                };

                // Add past KV cache if available (not first iteration)
                if (pastKVCache != null)
                {
                    for (int layer = 0; layer < numLayers; layer++)
                    {
                        inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.key", pastKVCache[$"present.{layer}.key"]));
                        inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.value", pastKVCache[$"present.{layer}.value"]));
                    }
                }
                else
                {
                    // First pass: provide empty KV cache tensors
                    // Shape: [batch_size, num_heads_kv, past_seq_len, head_dim]
                    // past_seq_len = 0 for first pass (no past context yet)
                    for (int layer = 0; layer < numLayers; layer++)
                    {
                        var emptyCache = new DenseTensor<float>(new[] { 1, numHeadsKV, 0, headDim });
                        inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.key", emptyCache));
                        inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.value", emptyCache));
                    }
                }

                // Run inference
                using var results = session.Run(inputs);
                
                // Extract logits and new KV cache
                var logitsOutput = results.FirstOrDefault(r => r.Name == "logits");
                if (logitsOutput == null)
                {
                    _logger.LogError("No logits output found in model results");
                    return "Error: Model output format unexpected.";
                }
                
                var outputTensor = logitsOutput.AsTensor<float>();
                var dimensions = outputTensor.Dimensions; // [batch, seq, vocab]
                int vocabSize = dimensions[2];
                
                // Get logits for last token in sequence
                int lastTokenIdx = outputTensor.Dimensions[1] - 1;
                
                // Find max logit (greedy decoding)
                int nextTokenId = 0;
                float maxLogit = float.MinValue;
                
                for (int v = 0; v < vocabSize; v++)
                {
                    float val = outputTensor[0, lastTokenIdx, v];
                    if (val > maxLogit)
                    {
                        maxLogit = val;
                        nextTokenId = v;
                    }
                }

                // Update KV cache for next iteration
                pastKVCache = new Dictionary<string, Tensor<float>>();
                for (int layer = 0; layer < numLayers; layer++)
                {
                    var keyOutput = results.FirstOrDefault(r => r.Name == $"present.{layer}.key");
                    var valueOutput = results.FirstOrDefault(r => r.Name == $"present.{layer}.value");
                    
                    if (keyOutput != null && valueOutput != null)
                    {
                        pastKVCache[$"present.{layer}.key"] = keyOutput.AsTensor<float>();
                        pastKVCache[$"present.{layer}.value"] = valueOutput.AsTensor<float>();
                    }
                }

                // Check for EOS token (1 for Gemma)
                if (nextTokenId == 1) // EOS token
                {
                    _logger.LogInformation("Hit EOS token at step {Step}", step);
                    break;
                }

                // Append to generated tokens
                generatedTokenIds.Add(nextTokenId);
                currentMask.Add(1);
                
                _logger.LogDebug("Generated token {TokenId} at step {Step}", nextTokenId, step);
            }

            // Decode generated tokens
            string generatedText = DecodeTokens(generatedTokenIds);
            
            _logger.LogInformation("Generated {TokenCount} tokens, decoded text length: {TextLength}", 
                generatedTokenIds.Count, generatedText.Length);
            
            return generatedText;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "LLM generation failed");
            return "Error generating answer.";
        }
        finally
        {
            ReleaseSession(session);
        }
    }
}

