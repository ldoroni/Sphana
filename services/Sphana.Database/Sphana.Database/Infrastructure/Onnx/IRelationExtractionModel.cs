namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// Interface for relation extraction model operations
/// </summary>
public interface IRelationExtractionModel
{
    /// <summary>
    /// Extracts relations from text given a list of entities
    /// </summary>
    /// <param name="text">The text to extract relations from</param>
    /// <param name="entities">List of entities found in the text</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of extracted relations</returns>
    Task<List<ExtractedRelation>> ExtractRelationsAsync(
        string text,
        List<ExtractedEntity> entities,
        CancellationToken cancellationToken = default);
}

