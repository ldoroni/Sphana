using Sphana.Database.Models.KnowledgeGraph;

namespace Sphana.Database.Tests.UnitTests.Models.KnowledgeGraph;

/// <summary>
/// Tests for Relation model
/// </summary>
public class RelationTests
{
    [Fact]
    public void Relation_Constructor_WithAllRequiredProperties_ShouldInitialize()
    {
        // Arrange & Act
        var relation = new Relation
        {
            Id = "r1",
            TenantId = "tenant1",
            IndexName = "index1",
            SourceEntityId = "e1",
            TargetEntityId = "e2",
            RelationType = "works_for",
            SourceChunkId = "chunk1"
        };

        // Assert
        relation.Should().NotBeNull();
        relation.Id.Should().Be("r1");
        relation.TenantId.Should().Be("tenant1");
        relation.IndexName.Should().Be("index1");
        relation.SourceEntityId.Should().Be("e1");
        relation.TargetEntityId.Should().Be("e2");
        relation.RelationType.Should().Be("works_for");
        relation.SourceChunkId.Should().Be("chunk1");
        relation.Properties.Should().NotBeNull();
        relation.Properties.Should().BeEmpty();
        relation.CreatedAt.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(5));
    }

    [Fact]
    public void Relation_WithConfidence_ShouldStoreConfidence()
    {
        // Arrange & Act
        var relation = new Relation
        {
            Id = "r1",
            TenantId = "tenant1",
            IndexName = "index1",
            SourceEntityId = "e1",
            TargetEntityId = "e2",
            RelationType = "works_for",
            SourceChunkId = "chunk1",
            Confidence = 0.9f
        };

        // Assert
        relation.Confidence.Should().Be(0.9f);
    }

    [Fact]
    public void Relation_WithProperties_ShouldStoreProperties()
    {
        // Arrange & Act
        var relation = new Relation
        {
            Id = "r1",
            TenantId = "tenant1",
            IndexName = "index1",
            SourceEntityId = "e1",
            TargetEntityId = "e2",
            RelationType = "works_for",
            SourceChunkId = "chunk1",
            Properties = new Dictionary<string, string>
            {
                ["since"] = "2020",
                ["department"] = "Engineering"
            }
        };

        // Assert
        relation.Properties.Should().HaveCount(2);
        relation.Properties["since"].Should().Be("2020");
        relation.Properties["department"].Should().Be("Engineering");
    }
}
