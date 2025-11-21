using Sphana.Database.Infrastructure.Onnx;

namespace Sphana.Database.Tests.Infrastructure.Onnx;

public class ExtractedRelationTests
{
    [Fact]
    public void ExtractedRelation_ShouldSetAndGetProperties()
    {
        // Arrange
        var sourceEntity = new ExtractedEntity
        {
            Text = "John Doe",
            Type = "PERSON",
            StartPosition = 0,
            EndPosition = 2
        };
        var targetEntity = new ExtractedEntity
        {
            Text = "Acme Corp",
            Type = "ORGANIZATION",
            StartPosition = 4,
            EndPosition = 6
        };
        var relationType = "WORKS_FOR";
        var confidence = 0.92f;

        // Act
        var relation = new ExtractedRelation
        {
            SourceEntity = sourceEntity,
            TargetEntity = targetEntity,
            RelationType = relationType,
            Confidence = confidence
        };

        // Assert
        relation.SourceEntity.Should().Be(sourceEntity);
        relation.TargetEntity.Should().Be(targetEntity);
        relation.RelationType.Should().Be(relationType);
        relation.Confidence.Should().Be(confidence);
    }

    [Fact]
    public void ExtractedRelation_SourceEntity_ShouldHaveCorrectProperties()
    {
        // Arrange
        var sourceEntity = new ExtractedEntity
        {
            Text = "John Doe",
            Type = "PERSON",
            StartPosition = 0,
            EndPosition = 2
        };

        var relation = new ExtractedRelation
        {
            SourceEntity = sourceEntity,
            TargetEntity = new ExtractedEntity { Text = "Acme", Type = "ORG", StartPosition = 3, EndPosition = 4 },
            RelationType = "WORKS_FOR",
            Confidence = 0.9f
        };

        // Assert
        relation.SourceEntity.Text.Should().Be("John Doe");
        relation.SourceEntity.Type.Should().Be("PERSON");
        relation.SourceEntity.StartPosition.Should().Be(0);
        relation.SourceEntity.EndPosition.Should().Be(2);
    }
}

