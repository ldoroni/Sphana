using System.Text.Json.Serialization.Metadata;
using Microsoft.AspNetCore.Builder;

namespace Sphana.RequestHandler;

/// <summary>
/// Extension methods for registering request handler infrastructure in the ASP.NET Core pipeline.
/// </summary>
public static class RequestHandlerExtensions
{
    /// <summary>
    /// Adds the managed exception handling middleware to the application pipeline.
    /// Requires a <see cref="JsonTypeInfo{ErrorResponse}"/> for AOT-safe JSON serialization.
    /// </summary>
    public static IApplicationBuilder UseManagedExceptionHandler(
        this IApplicationBuilder applicationBuilder,
        JsonTypeInfo<ErrorResponse> errorResponseTypeInfo)
    {
        return applicationBuilder.UseMiddleware<ManagedExceptionMiddleware>(errorResponseTypeInfo);
    }
}