using Sphana.Database.Infrastructure.Onnx;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// Interface for Named Entity Recognition (NER) model
/// </summary>
public interface INerModel
{
    /// <summary>
    /// Extract entities from text
    /// </summary>
    Task<List<ExtractedEntity>> ExtractEntitiesAsync(string text, CancellationToken cancellationToken = default);
}

