using Sphana.Database.Services;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// Interface for LLM answer generation
/// </summary>
public interface ILlmGeneratorModel
{
    /// <summary>
    /// Generate answer from query and context
    /// </summary>
    Task<string> GenerateAnswerAsync(string prompt, int maxTokens, CancellationToken cancellationToken = default);
}

