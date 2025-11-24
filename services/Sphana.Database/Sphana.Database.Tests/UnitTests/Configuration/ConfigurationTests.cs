using Sphana.Database.Configuration;

namespace Sphana.Database.Tests.UnitTests.Configuration;

public class ConfigurationTests
{
    [Fact]
    public void SphanaConfiguration_Should_Have_Default_Values()
    {
        // Arrange & Act
        var config = new SphanaConfiguration();

        // Assert
        config.Should().NotBeNull();
        config.Models.Should().NotBeNull();
        config.VectorIndex.Should().NotBeNull();
        config.KnowledgeGraph.Should().NotBeNull();
        config.Cache.Should().NotBeNull();
        config.Ingestion.Should().NotBeNull();
        config.Query.Should().NotBeNull();
    }

    [Fact]
    public void OnnxModelConfiguration_Should_Have_Valid_Defaults()
    {
        // Arrange & Act
        var config = new OnnxModelConfiguration();

        // Assert
        config.EmbeddingDimension.Should().BeGreaterThan(0);
        config.BatchSize.Should().BeGreaterThan(0);
        config.MaxBatchWaitMs.Should().BeGreaterThan(0);
        config.IntraOpNumThreads.Should().BeGreaterThan(0);
        config.InterOpNumThreads.Should().BeGreaterThan(0);
    }

    [Fact]
    public void VectorIndexConfiguration_Should_Have_Valid_Defaults()
    {
        // Arrange & Act
        var config = new VectorIndexConfiguration();

        // Assert
        config.Dimension.Should().BeGreaterThan(0);
        config.HnswM.Should().BeGreaterThan(0);
        config.HnswEfConstruction.Should().BeGreaterThan(0);
        config.HnswEfSearch.Should().BeGreaterThan(0);
        config.MaxResults.Should().BeGreaterThan(0);
    }

    [Fact]
    public void KnowledgeGraphConfiguration_Should_Have_Valid_Defaults()
    {
        // Arrange & Act
        var config = new KnowledgeGraphConfiguration();

        // Assert
        config.MaxTraversalDepth.Should().BeGreaterThan(0);
        config.PcsrSlackRatio.Should().BeGreaterThan(0);
        config.PcsrSlackRatio.Should().BeLessThanOrEqualTo(1);
        config.BlockSize.Should().BeGreaterThan(0);
    }

    [Fact]
    public void CacheConfiguration_Should_Have_Valid_Defaults()
    {
        // Arrange & Act
        var config = new CacheConfiguration();

        // Assert
        config.InMemoryCacheSizeMb.Should().BeGreaterThan(0);
        config.ExpirationMinutes.Should().BeGreaterThan(0);
    }

    [Fact]
    public void IngestionConfiguration_Should_Have_Valid_Defaults()
    {
        // Arrange & Act
        var config = new IngestionConfiguration();

        // Assert
        config.ChunkSize.Should().BeGreaterThan(0);
        config.ChunkOverlap.Should().BeGreaterThanOrEqualTo(0);
        config.MaxConcurrency.Should().BeGreaterThan(0);
        config.BatchSize.Should().BeGreaterThan(0);
        config.MinRelationConfidence.Should().BeGreaterThanOrEqualTo(0);
        config.MinRelationConfidence.Should().BeLessThanOrEqualTo(1);
    }

    [Fact]
    public void QueryConfiguration_Should_Have_Valid_Defaults()
    {
        // Arrange & Act
        var config = new QueryConfiguration();

        // Assert
        config.TargetP95LatencyMs.Should().BeGreaterThan(0);
        config.VectorSearchTopK.Should().BeGreaterThan(0);
        config.MaxSubgraphs.Should().BeGreaterThan(0);
        config.MaxGenerationTokens.Should().BeGreaterThan(0);
        config.VectorSearchWeight.Should().BeGreaterThanOrEqualTo(0);
        config.VectorSearchWeight.Should().BeLessThanOrEqualTo(1);
        config.GraphSearchWeight.Should().BeGreaterThanOrEqualTo(0);
        config.GraphSearchWeight.Should().BeLessThanOrEqualTo(1);
    }

    [Fact]
    public void QueryConfiguration_Weights_Should_Sum_To_One()
    {
        // Arrange
        var config = new QueryConfiguration();

        // Act
        var sum = config.VectorSearchWeight + config.GraphSearchWeight;

        // Assert
        sum.Should().BeApproximately(1.0f, 0.01f);
    }
}

