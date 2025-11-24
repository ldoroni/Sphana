using Sphana.Database.Models.KnowledgeGraph;

namespace Sphana.Database.Tests.UnitTests.Models.KnowledgeGraph;

/// <summary>
/// Tests for KnowledgeSubgraph model
/// </summary>
public class KnowledgeSubgraphTests
{
    [Fact]
    public void KnowledgeSubgraph_Constructor_WithAllRequiredProperties_ShouldInitialize()
    {
        // Arrange & Act
        var subgraph = new KnowledgeSubgraph
        {
            Id = "sg1",
            Entities = new List<Entity>(),
            Relations = new List<Relation>()
        };

        // Assert
        subgraph.Should().NotBeNull();
        subgraph.Id.Should().Be("sg1");
        subgraph.Entities.Should().NotBeNull();
        subgraph.Entities.Should().BeEmpty();
        subgraph.Relations.Should().NotBeNull();
        subgraph.Relations.Should().BeEmpty();
    }

    [Fact]
    public void KnowledgeSubgraph_WithEntities_ShouldStoreEntities()
    {
        // Arrange
        var entities = new List<Entity>
        {
            new Entity
            {
                Id = "e1",
                TenantId = "tenant1",
                IndexName = "index1",
                Text = "John Smith",
                Type = "PERSON",
                SourceChunkId = "chunk1"
            },
            new Entity
            {
                Id = "e2",
                TenantId = "tenant1",
                IndexName = "index1",
                Text = "Microsoft",
                Type = "ORGANIZATION",
                SourceChunkId = "chunk1"
            }
        };

        // Act
        var subgraph = new KnowledgeSubgraph
        {
            Id = "sg1",
            Entities = entities,
            Relations = new List<Relation>()
        };

        // Assert
        subgraph.Entities.Should().HaveCount(2);
        subgraph.Entities[0].Text.Should().Be("John Smith");
        subgraph.Entities[1].Text.Should().Be("Microsoft");
    }

    [Fact]
    public void KnowledgeSubgraph_WithRelations_ShouldStoreRelations()
    {
        // Arrange
        var relations = new List<Relation>
        {
            new Relation
            {
                Id = "r1",
                TenantId = "tenant1",
                IndexName = "index1",
                SourceEntityId = "e1",
                TargetEntityId = "e2",
                RelationType = "works_for",
                SourceChunkId = "chunk1"
            }
        };

        // Act
        var subgraph = new KnowledgeSubgraph
        {
            Id = "sg1",
            Entities = new List<Entity>(),
            Relations = relations
        };

        // Assert
        subgraph.Relations.Should().HaveCount(1);
        subgraph.Relations[0].RelationType.Should().Be("works_for");
    }

    [Fact]
    public void KnowledgeSubgraph_WithRelevanceScore_ShouldStoreScore()
    {
        // Arrange & Act
        var subgraph = new KnowledgeSubgraph
        {
            Id = "sg1",
            Entities = new List<Entity>(),
            Relations = new List<Relation>(),
            RelevanceScore = 0.85f
        };

        // Assert
        subgraph.RelevanceScore.Should().Be(0.85f);
    }
}
