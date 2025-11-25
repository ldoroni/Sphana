using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Logging.Console;
using System.Text;

namespace Sphana.Database.Infrastructure.Logging;

/// <summary>
/// Custom console formatter that outputs logs in the format:
/// timestamp {threadname} LOGLEVEL: [class] logmessage
/// </summary>
public sealed class SphanaConsoleFormatter : ConsoleFormatter
{
    public SphanaConsoleFormatter() : base("sphana")
    {
    }

    public override void Write<TState>(
        in LogEntry<TState> logEntry,
        IExternalScopeProvider? scopeProvider,
        TextWriter textWriter)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss.fff");
        var threadName = Thread.CurrentThread.Name ?? Thread.CurrentThread.ManagedThreadId.ToString();
        var logLevel = logEntry.LogLevel.ToString().ToUpperInvariant();
        var category = logEntry.Category;
        var message = logEntry.Formatter(logEntry.State, logEntry.Exception);

        // Format: timestamp {threadname} LOGLEVEL: [class] logmessage
        var logLine = $"{timestamp} {{{threadName}}} {logLevel}: [{category}] {message}";
        textWriter.WriteLine(logLine);

        // Write exception if present
        if (logEntry.Exception != null)
        {
            textWriter.WriteLine(logEntry.Exception.ToString());
        }
    }
}

