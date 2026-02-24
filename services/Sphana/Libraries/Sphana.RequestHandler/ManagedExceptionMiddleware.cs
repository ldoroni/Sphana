using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using ManagedExceptionType = Sphana.ManagedException.ManagedException;
using Sphana.ManagedException;

namespace Sphana.RequestHandler;

/// <summary>
/// ASP.NET Core middleware that catches ManagedException instances and returns
/// structured error responses with appropriate HTTP status codes.
/// </summary>
public sealed class ManagedExceptionMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<ManagedExceptionMiddleware> _logger;
    private readonly JsonTypeInfo<ErrorResponse> _errorResponseTypeInfo;

    public ManagedExceptionMiddleware(
        RequestDelegate next,
        ILogger<ManagedExceptionMiddleware> logger,
        JsonTypeInfo<ErrorResponse> errorResponseTypeInfo)
    {
        _next = next;
        _logger = logger;
        _errorResponseTypeInfo = errorResponseTypeInfo;
    }

    public async Task InvokeAsync(HttpContext httpContext)
    {
        try
        {
            await _next(httpContext);
        }
        catch (ManagedExceptionType managedException)
        {
            _logger.LogWarning(
                managedException,
                "Managed exception occurred: {DiagnosticCode} - {Message}",
                managedException.ErrorDetails.DiagnosticCode,
                managedException.ErrorDetails.Message);

            await WriteErrorResponse(httpContext, managedException.ErrorDetails);
        }
        catch (Exception unhandledException)
        {
            _logger.LogError(
                unhandledException,
                "Unhandled exception occurred: {Message}",
                unhandledException.Message);

            var errorDetails = new ErrorDetails(
                HttpStatusCode.InternalServerError,
                "INTERNAL_ERROR",
                new Dictionary<string, string>(),
                "An unexpected internal error occurred.");

            await WriteErrorResponse(httpContext, errorDetails);
        }
    }

    private async Task WriteErrorResponse(HttpContext httpContext, ErrorDetails errorDetails)
    {
        httpContext.Response.StatusCode = (int)errorDetails.StatusCode;
        httpContext.Response.ContentType = "application/json";

        var errorResponse = new ErrorResponse
        {
            DiagnosticCode = errorDetails.DiagnosticCode,
            Message = errorDetails.Message,
            DiagnosticDetails = errorDetails.DiagnosticDetails
        };

        await httpContext.Response.WriteAsync(
            JsonSerializer.Serialize(errorResponse, _errorResponseTypeInfo));
    }
}