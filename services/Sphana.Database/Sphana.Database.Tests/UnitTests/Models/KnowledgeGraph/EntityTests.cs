using Sphana.Database.Models.KnowledgeGraph;

namespace Sphana.Database.Tests.UnitTests.Models.KnowledgeGraph;

/// <summary>
/// Tests for Entity model
/// </summary>
public class EntityTests
{
    [Fact]
    public void Entity_Constructor_WithAllRequiredProperties_ShouldInitialize()
    {
        // Arrange & Act
        var entity = new Entity
        {
            Id = "e1",
            TenantId = "tenant1",
            IndexName = "index1",
            Text = "John Smith",
            Type = "PERSON",
            SourceChunkId = "chunk1"
        };

        // Assert
        entity.Should().NotBeNull();
        entity.Id.Should().Be("e1");
        entity.TenantId.Should().Be("tenant1");
        entity.IndexName.Should().Be("index1");
        entity.Text.Should().Be("John Smith");
        entity.Type.Should().Be("PERSON");
        entity.SourceChunkId.Should().Be("chunk1");
        entity.Properties.Should().NotBeNull();
        entity.Properties.Should().BeEmpty();
        entity.CreatedAt.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(5));
    }

    [Fact]
    public void Entity_WithEmbedding_ShouldStoreEmbedding()
    {
        // Arrange & Act
        var embedding = new float[] { 0.1f, 0.2f, 0.3f };
        var entity = new Entity
        {
            Id = "e1",
            TenantId = "tenant1",
            IndexName = "index1",
            Text = "John Smith",
            Type = "PERSON",
            SourceChunkId = "chunk1",
            Embedding = embedding
        };

        // Assert
        entity.Embedding.Should().NotBeNull();
        entity.Embedding.Should().HaveCount(3);
        entity.Embedding.Should().BeEquivalentTo(embedding);
    }

    [Fact]
    public void Entity_WithQuantizedEmbedding_ShouldStoreQuantizedEmbedding()
    {
        // Arrange & Act
        var quantizedEmbedding = new byte[] { 128, 64, 32 };
        var entity = new Entity
        {
            Id = "e1",
            TenantId = "tenant1",
            IndexName = "index1",
            Text = "John Smith",
            Type = "PERSON",
            SourceChunkId = "chunk1",
            QuantizedEmbedding = quantizedEmbedding
        };

        // Assert
        entity.QuantizedEmbedding.Should().NotBeNull();
        entity.QuantizedEmbedding.Should().HaveCount(3);
        entity.QuantizedEmbedding.Should().BeEquivalentTo(quantizedEmbedding);
    }

    [Fact]
    public void Entity_WithProperties_ShouldStoreProperties()
    {
        // Arrange & Act
        var entity = new Entity
        {
            Id = "e1",
            TenantId = "tenant1",
            IndexName = "index1",
            Text = "John Smith",
            Type = "PERSON",
            SourceChunkId = "chunk1",
            Properties = new Dictionary<string, string>
            {
                ["age"] = "30",
                ["occupation"] = "Engineer"
            }
        };

        // Assert
        entity.Properties.Should().HaveCount(2);
        entity.Properties["age"].Should().Be("30");
        entity.Properties["occupation"].Should().Be("Engineer");
    }
}
