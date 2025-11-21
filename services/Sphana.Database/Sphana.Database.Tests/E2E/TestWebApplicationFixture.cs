using Grpc.Net.Client;
using Sphana.Database.RPC.V1;

namespace Sphana.Database.Tests.E2E;

/// <summary>
/// Shared fixture for E2E tests to avoid recreating the web application factory for each test
/// </summary>
public class TestWebApplicationFixture : IAsyncLifetime
{
    public TestWebApplicationFactory? Factory { get; private set; }
    public GrpcChannel? Channel { get; private set; }
    public SphanaDatabase.SphanaDatabaseClient? Client { get; private set; }

    public async Task InitializeAsync()
    {
        Factory = new TestWebApplicationFactory();
        var httpClient = Factory.CreateDefaultClient();
        Channel = GrpcChannel.ForAddress(httpClient.BaseAddress!, new GrpcChannelOptions
        {
            HttpClient = httpClient
        });
        Client = new SphanaDatabase.SphanaDatabaseClient(Channel);
        
        await Task.CompletedTask;
    }

    public async Task DisposeAsync()
    {
        Channel?.Dispose();
        Factory?.Dispose();
        await Task.CompletedTask;
    }
}

