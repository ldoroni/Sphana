namespace Sphana.Database.Configuration;

/// <summary>
/// Configuration for debug logging features
/// </summary>
public sealed class DebugConfiguration
{
    /// <summary>
    /// Master switch for all debug logging
    /// </summary>
    public bool EnableVerboseLogging { get; set; } = false;

    /// <summary>
    /// Log decoded text of tokenized prompts in LLM generation
    /// </summary>
    public bool LogTokenizedText { get; set; } = false;

    /// <summary>
    /// Log detailed ingestion data (chunks, entities, relations, embeddings)
    /// </summary>
    public bool LogIngestionData { get; set; } = false;
}

