namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// Interface for embedding model operations
/// </summary>
public interface IEmbeddingModel
{
    /// <summary>
    /// Generates embeddings for an array of texts
    /// </summary>
    /// <param name="texts">Array of texts to embed</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Array of embedding vectors</returns>
    Task<float[][]> GenerateEmbeddingsAsync(string[] texts, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates an embedding for a single text
    /// </summary>
    /// <param name="text">Text to embed</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Embedding vector</returns>
    Task<float[]> GenerateEmbeddingAsync(string text, CancellationToken cancellationToken = default);
}

