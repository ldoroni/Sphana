using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading.Channels;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// ONNX model wrapper for embedding generation with batching support
/// </summary>
public sealed class EmbeddingModel : OnnxModelBase, IEmbeddingModel
{
    private readonly Channel<EmbeddingRequest> _batchChannel;
    private readonly int _embeddingDimension;
    private readonly int _maxBatchSize;
    private readonly int _maxBatchWaitMs;
    private readonly Task _batchProcessingTask;
    private readonly CancellationTokenSource _cancellationTokenSource;
    private readonly UncasedTokenizer _tokenizer;

    public EmbeddingModel(
        string modelPath,
        int embeddingDimension,
        bool useGpu,
        int gpuDeviceId,
        int maxPoolSize,
        int maxBatchSize,
        int maxBatchWaitMs,
        UncasedTokenizer tokenizer,
        ILogger<EmbeddingModel> logger)
        : base(modelPath, useGpu, gpuDeviceId, maxPoolSize, logger)
    {
        _tokenizer = tokenizer;
        _embeddingDimension = embeddingDimension;
        _maxBatchSize = maxBatchSize;
        _maxBatchWaitMs = maxBatchWaitMs;

        _batchChannel = Channel.CreateUnbounded<EmbeddingRequest>();
        _cancellationTokenSource = new CancellationTokenSource();
        _batchProcessingTask = Task.Run(() => ProcessBatchesAsync(_cancellationTokenSource.Token));
    }

    /// <summary>
    /// Generate embeddings for a batch of texts
    /// </summary>
    public async Task<float[][]> GenerateEmbeddingsAsync(
        string[] texts, 
        CancellationToken cancellationToken = default)
    {
        if (texts == null || texts.Length == 0)
        {
            return Array.Empty<float[]>();
        }

        var tcs = new TaskCompletionSource<float[][]>();
        var request = new EmbeddingRequest(texts, tcs);

        await _batchChannel.Writer.WriteAsync(request, cancellationToken);

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(TimeSpan.FromSeconds(30)); // Timeout

        return await tcs.Task.WaitAsync(cts.Token);
    }

    /// <summary>
    /// Generate a single embedding
    /// </summary>
    public async Task<float[]> GenerateEmbeddingAsync(
        string text, 
        CancellationToken cancellationToken = default)
    {
        var results = await GenerateEmbeddingsAsync(new[] { text }, cancellationToken);
        return results[0];
    }

    /// <summary>
    /// Normalize embeddings to unit vectors (for cosine similarity via dot product)
    /// </summary>
    public static float[] Normalize(float[] embedding)
    {
        var magnitude = Math.Sqrt(embedding.Sum(x => x * x));
        if (magnitude == 0) return embedding;

        var normalized = new float[embedding.Length];
        for (int i = 0; i < embedding.Length; i++)
        {
            normalized[i] = (float)(embedding[i] / magnitude);
        }
        return normalized;
    }

    /// <summary>
    /// Quantize float embeddings to int8 for storage efficiency
    /// </summary>
    public static byte[] Quantize(float[] embedding)
    {
        // Simple min-max quantization to int8
        var min = embedding.Min();
        var max = embedding.Max();
        var scale = (max - min) / 255.0f;

        var quantized = new byte[embedding.Length];
        for (int i = 0; i < embedding.Length; i++)
        {
            quantized[i] = (byte)Math.Clamp((embedding[i] - min) / scale, 0, 255);
        }
        return quantized;
    }

    /// <summary>
    /// Dequantize int8 embeddings back to float
    /// </summary>
    public static float[] Dequantize(byte[] quantized, float min, float max)
    {
        var scale = (max - min) / 255.0f;
        var embedding = new float[quantized.Length];
        for (int i = 0; i < quantized.Length; i++)
        {
            embedding[i] = quantized[i] * scale + min;
        }
        return embedding;
    }

    private async Task ProcessBatchesAsync(CancellationToken cancellationToken)
    {
        var currentBatch = new List<EmbeddingRequest>();

        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                // Wait for first request or timeout
                var hasRequest = await _batchChannel.Reader.WaitToReadAsync(cancellationToken);
                if (!hasRequest) continue;

                currentBatch.Clear();

                // Collect requests for batching
                var deadline = DateTime.UtcNow.AddMilliseconds(_maxBatchWaitMs);
                while (currentBatch.Count < _maxBatchSize && 
                       DateTime.UtcNow < deadline &&
                       _batchChannel.Reader.TryRead(out var request))
                {
                    currentBatch.Add(request);
                }

                if (currentBatch.Count > 0)
                {
                    await ProcessBatchAsync(currentBatch, cancellationToken);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing embedding batch");
            }
        }
    }

    private async Task ProcessBatchAsync(
        List<EmbeddingRequest> batch, 
        CancellationToken cancellationToken)
    {
        try
        {
            // Flatten all texts from all requests
            var allTexts = batch.SelectMany(r => r.Texts).ToArray();
            
            // Run inference
            var embeddings = await RunInferenceAsync(allTexts, cancellationToken);

            // Split results back to requests
            int offset = 0;
            foreach (var request in batch)
            {
                var count = request.Texts.Length;
                var results = embeddings.Skip(offset).Take(count).ToArray();
                offset += count;

                request.CompletionSource.SetResult(results);
            }
        }
        catch (Exception ex)
        {
            // Fail all requests in the batch
            foreach (var request in batch)
            {
                request.CompletionSource.SetException(ex);
            }
        }
    }

    private async Task<float[][]> RunInferenceAsync(
        string[] texts, 
        CancellationToken cancellationToken)
    {
        var session = await AcquireSessionAsync(cancellationToken);
        try
        {
            var inputTensors = PrepareInputTensors(texts);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensors.InputIds),
                NamedOnnxValue.CreateFromTensor("attention_mask", inputTensors.AttentionMask)
            };

            if (session.InputMetadata.ContainsKey("token_type_ids"))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", inputTensors.TokenTypeIds));
            }

            using var results = session.Run(inputs);
            
            // Prefer fetching by name if possible, otherwise take first
            var outputTensor = results.FirstOrDefault(r => r.Name == "embeddings") ?? results.First();
            var output = outputTensor.AsEnumerable<float>().ToArray();

            // Reshape output to [batch_size, embedding_dim]
            var embeddings = new float[texts.Length][];
            for (int i = 0; i < texts.Length; i++)
            {
                embeddings[i] = new float[_embeddingDimension];
                Array.Copy(output, i * _embeddingDimension, embeddings[i], 0, _embeddingDimension);
                
                // Normalize embedding (even if model does it, safe to do again)
                embeddings[i] = Normalize(embeddings[i]);
            }

            return embeddings;
        }
        finally
        {
            ReleaseSession(session);
        }
    }

    private (Tensor<long> InputIds, Tensor<long> AttentionMask, Tensor<long> TokenTypeIds) PrepareInputTensors(string[] texts)
    {
        const int maxLength = 512;
        
        var inputIds = new long[texts.Length][];
        var attentionMask = new long[texts.Length][];
        var tokenTypeIds = new long[texts.Length][];

        for (int i = 0; i < texts.Length; i++)
        {
            // Tokenize text using BERT tokenizer
            var tokens = _tokenizer.Tokenize(texts[i]);
            var encoded = _tokenizer.Encode(tokens.Count() + 2, texts[i]); // +2 for [CLS] and [SEP]
            
            // Encode returns List<(long InputIds, long TokenTypeIds, long AttentionMask)>
            inputIds[i] = new long[maxLength];
            attentionMask[i] = new long[maxLength];
            tokenTypeIds[i] = new long[maxLength];
            
            // Copy the encoded tokens (truncate if necessary)
            var lengthToCopy = Math.Min(encoded.Count, maxLength);
            for (int j = 0; j < lengthToCopy; j++)
            {
                inputIds[i][j] = encoded[j].InputIds;
                attentionMask[i][j] = encoded[j].AttentionMask;
                tokenTypeIds[i][j] = encoded[j].TokenTypeIds;
            }
            
            // Padding is already 0 from initialization
        }

        var inputIdsTensor = CreateTensor(inputIds, new[] { texts.Length, maxLength });
        var attentionMaskTensor = CreateTensor(attentionMask, new[] { texts.Length, maxLength });
        var tokenTypeIdsTensor = CreateTensor(tokenTypeIds, new[] { texts.Length, maxLength });

        return (inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor);
    }

    public override void Dispose()
    {
        _cancellationTokenSource?.Cancel();
        _batchProcessingTask?.Wait(TimeSpan.FromSeconds(5));
        _batchChannel?.Writer.Complete();
        _cancellationTokenSource?.Dispose();
        base.Dispose();
    }

    private record EmbeddingRequest(
        string[] Texts, 
        TaskCompletionSource<float[][]> CompletionSource);
}
