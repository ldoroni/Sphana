namespace Sphana.Database.Configuration;

/// <summary>
/// Configuration for ONNX models
/// </summary>
public sealed class OnnxModelConfiguration
{
    /// <summary>
    /// Path to the embedding model ONNX file
    /// </summary>
    public string EmbeddingModelPath { get; set; } = "models/embedding.onnx";

    /// <summary>
    /// Path to the relation extraction model ONNX file
    /// </summary>
    public string RelationExtractionModelPath { get; set; } = "models/relation_extraction.onnx";

    /// <summary>
    /// Path to the NER model ONNX file
    /// </summary>
    public string NerModelPath { get; set; } = "models/ner.onnx";

    /// <summary>
    /// Path to the GNN ranker model ONNX file
    /// </summary>
    public string GnnRankerModelPath { get; set; } = "models/gnn_ranker.onnx";

    /// <summary>
    /// Path to the LLM generator model ONNX file
    /// </summary>
    public string LlmGeneratorModelPath { get; set; } = "models/llm_generator.onnx";

    /// <summary>
    /// Path to the LLM tokenizer file (tokenizer.json)
    /// </summary>
    public string LlmTokenizerPath { get; set; } = "models/tokenizer.json";

    /// <summary>
    /// Path to the tokenizer vocabularies file
    /// </summary>
    public string VocabulariesPath { get; set; } = "Vocabularies/base_uncased.txt";

    /// <summary>
    /// Embedding dimension
    /// </summary>
    public int EmbeddingDimension { get; set; } = 384;

    /// <summary>
    /// Use GPU for inference (CUDA execution provider)
    /// </summary>
    public bool UseGpu { get; set; } = true;

    /// <summary>
    /// GPU device ID
    /// </summary>
    public int GpuDeviceId { get; set; } = 0;

    /// <summary>
    /// CUDA compute capability (e.g., 8.0 for A100)
    /// </summary>
    public string CudaComputeCapability { get; set; } = "8.0";

    /// <summary>
    /// Batch size for inference
    /// </summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Maximum batch wait time in milliseconds
    /// </summary>
    public int MaxBatchWaitMs { get; set; } = 5;

    /// <summary>
    /// Number of inference threads for CPU fallback
    /// </summary>
    public int IntraOpNumThreads { get; set; } = 4;

    /// <summary>
    /// Number of parallel inferences
    /// </summary>
    public int InterOpNumThreads { get; set; } = 4;
}

