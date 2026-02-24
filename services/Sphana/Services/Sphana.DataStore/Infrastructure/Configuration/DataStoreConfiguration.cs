namespace Sphana.DataStore.Infrastructure.Configuration;

/// <summary>
/// Strongly-typed configuration for the data store service.
/// </summary>
public sealed class DataStoreConfiguration
{
    public const string SectionName = "DataStore";

    public string DatabasePath { get; set; } = ".database";
    public string DefaultShardName { get; set; } = "default";
    public int PrometheusPort { get; set; } = 9090;
}