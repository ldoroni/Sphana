namespace Sphana.Database.Models;

public class CombinedResult
{
    public required string ChunkId { get; set; }
    public float VectorScore { get; set; }
    public float GraphScore { get; set; }
    public float CombinedScore { get; set; }
}

