using Sphana.Database.Models;

namespace Sphana.Database.Tests.UnitTests.Models;

/// <summary>
/// Tests for Document model
/// </summary>
public class DocumentTests
{
    [Fact]
    public void Document_Constructor_WithAllRequiredProperties_ShouldInitialize()
    {
        // Arrange & Act
        var document = new Document
        {
            Id = "doc1",
            TenantId = "tenant1",
            IndexName = "index1",
            Title = "Test Document",
            Content = "This is test content"
        };

        // Assert
        document.Should().NotBeNull();
        document.Id.Should().Be("doc1");
        document.TenantId.Should().Be("tenant1");
        document.IndexName.Should().Be("index1");
        document.Title.Should().Be("Test Document");
        document.Content.Should().Be("This is test content");
        document.Metadata.Should().NotBeNull();
        document.Metadata.Should().BeEmpty();
        document.IndexedAt.Should().BeCloseTo(DateTime.UtcNow, TimeSpan.FromSeconds(5));
    }

    [Fact]
    public void Document_WithMetadata_ShouldStoreMetadata()
    {
        // Arrange & Act
        var document = new Document
        {
            Id = "doc1",
            TenantId = "tenant1",
            IndexName = "index1",
            Title = "Test Document",
            Content = "This is test content",
            Metadata = new Dictionary<string, string>
            {
                ["author"] = "John Doe",
                ["category"] = "Science"
            }
        };

        // Assert
        document.Metadata.Should().HaveCount(2);
        document.Metadata["author"].Should().Be("John Doe");
        document.Metadata["category"].Should().Be("Science");
    }

    [Fact]
    public void Document_WithContentHash_ShouldStoreHash()
    {
        // Arrange & Act
        var document = new Document
        {
            Id = "doc1",
            TenantId = "tenant1",
            IndexName = "index1",
            Title = "Test Document",
            Content = "This is test content",
            ContentHash = "abc123hash"
        };

        // Assert
        document.ContentHash.Should().Be("abc123hash");
    }

    [Fact]
    public void Document_WithEmptyContent_ShouldBeValid()
    {
        // Arrange & Act
        var document = new Document
        {
            Id = "doc1",
            TenantId = "tenant1",
            IndexName = "index1",
            Title = "Empty Document",
            Content = ""
        };

        // Assert
        document.Content.Should().BeEmpty();
    }

    [Fact]
    public void Document_WithLongContent_ShouldStoreCorrectly()
    {
        // Arrange
        var longContent = string.Join(" ", Enumerable.Repeat("word", 10000));

        // Act
        var document = new Document
        {
            Id = "doc1",
            TenantId = "tenant1",
            IndexName = "index1",
            Title = "Long Document",
            Content = longContent
        };

        // Assert
        document.Content.Length.Should().BeGreaterThan(40000);
    }
}
