using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using Microsoft.Extensions.Logging;
using Sphana.DataStore.Domain.Interfaces;
using Sphana.ManagedException.Internal;

namespace Sphana.DataStore.Infrastructure.Persistence.RocksDb;

/// <summary>
/// Generic RocksDB-backed document repository using column families for table isolation.
/// Uses direct P/Invoke into the RocksDB C library for AOT compatibility.
/// </summary>
public class RocksDbDocumentRepository<TDocument> : IDocumentRepository<TDocument>, IDisposable
    where TDocument : class
{
    private readonly string _databasePath;
    private readonly ILogger _logger;
    private readonly JsonTypeInfo<TDocument> _jsonTypeInfo;
    private readonly object _lockObject = new();

    private nint _dbHandle;
    private nint _writeOptions;
    private nint _readOptions;
    private readonly Dictionary<string, nint> _columnFamilyHandles = new();
    private bool _disposed;

    public RocksDbDocumentRepository(string databasePath, ILogger logger, JsonTypeInfo<TDocument> jsonTypeInfo)
    {
        _databasePath = databasePath;
        _logger = logger;
        _jsonTypeInfo = jsonTypeInfo;
        Directory.CreateDirectory(_databasePath);
        OpenDatabase();
    }

    private void OpenDatabase()
    {
        var options = RocksDbNativeMethods.OptionsCreate();
        try
        {
            RocksDbNativeMethods.OptionsSetCreateIfMissing(options, 1);
            RocksDbNativeMethods.OptionsSetCreateMissingColumnFamilies(options, 1);

            // Discover existing column families or use default
            var familyNames = GetExistingColumnFamilies(options);

            if (familyNames.Count == 0)
                familyNames.Add("default");

            // Prepare arrays for rocksdb_open_column_families
            var count = familyNames.Count;
            var cfNamePtrs = new nint[count];
            var cfOptionPtrs = new nint[count];
            var cfHandles = new nint[count];

            // Create per-CF options (all use default settings)
            for (var i = 0; i < count; i++)
            {
                cfNamePtrs[i] = MarshalUtf8String(familyNames[i]);
                cfOptionPtrs[i] = RocksDbNativeMethods.OptionsCreate();
            }

            try
            {
                _dbHandle = RocksDbNativeMethods.OpenColumnFamilies(
                    options, _databasePath, count, cfNamePtrs, cfOptionPtrs, cfHandles, out var errPtr);
                ThrowOnError(errPtr, "Failed to open RocksDB database");

                // Cache column family handles
                for (var i = 0; i < count; i++)
                {
                    _columnFamilyHandles[familyNames[i]] = cfHandles[i];
                }
            }
            finally
            {
                // Free the temporary name strings and per-CF options
                for (var i = 0; i < count; i++)
                {
                    Marshal.FreeHGlobal(cfNamePtrs[i]);
                    RocksDbNativeMethods.OptionsDestroy(cfOptionPtrs[i]);
                }
            }

            // Create shared read/write options
            _writeOptions = RocksDbNativeMethods.WriteOptionsCreate();
            _readOptions = RocksDbNativeMethods.ReadOptionsCreate();
        }
        finally
        {
            RocksDbNativeMethods.OptionsDestroy(options);
        }
    }

    /// <summary>
    /// List existing column families for the database path. Returns empty if DB doesn't exist yet.
    /// </summary>
    private List<string> GetExistingColumnFamilies(nint options)
    {
        var families = new List<string>();

        if (!Directory.Exists(_databasePath) || Directory.GetFiles(_databasePath).Length == 0)
            return families;

        nint listPtr;
        nuint count;
        nint errPtr;

        try
        {
            listPtr = RocksDbNativeMethods.ListColumnFamilies(options, _databasePath, out count, out errPtr);
        }
        catch
        {
            return families;
        }

        if (errPtr != nint.Zero)
        {
            // Database doesn't exist yet or is corrupted — just return empty
            RocksDbNativeMethods.Free(errPtr);
            return families;
        }

        try
        {
            for (nuint i = 0; i < count; i++)
            {
                var strPtr = Marshal.ReadIntPtr(listPtr + (nint)((uint)i * (uint)nint.Size));
                var name = Marshal.PtrToStringUTF8(strPtr);
                if (name is not null)
                    families.Add(name);
            }
        }
        finally
        {
            RocksDbNativeMethods.ListColumnFamiliesDestroy(listPtr, count);
        }

        return families;
    }

    public void InitializeTable(string tableName)
    {
        lock (_lockObject)
        {
            if (_columnFamilyHandles.ContainsKey(tableName))
                return;

            var cfOptions = RocksDbNativeMethods.OptionsCreate();
            try
            {
                var handle = RocksDbNativeMethods.CreateColumnFamily(
                    _dbHandle, cfOptions, tableName, out var errPtr);
                ThrowOnError(errPtr, $"Failed to create column family '{tableName}'");

                _columnFamilyHandles[tableName] = handle;
                _logger.LogInformation("Initialized RocksDB table '{TableName}' at '{DatabasePath}'",
                    tableName, _databasePath);
            }
            finally
            {
                RocksDbNativeMethods.OptionsDestroy(cfOptions);
            }
        }
    }

    public void DropTable(string tableName)
    {
        lock (_lockObject)
        {
            if (!_columnFamilyHandles.TryGetValue(tableName, out var handle))
                return;

            RocksDbNativeMethods.DropColumnFamily(_dbHandle, handle, out var errPtr);
            ThrowOnError(errPtr, $"Failed to drop column family '{tableName}'");

            RocksDbNativeMethods.ColumnFamilyHandleDestroy(handle);
            _columnFamilyHandles.Remove(tableName);

            _logger.LogInformation("Dropped RocksDB table '{TableName}' at '{DatabasePath}'",
                tableName, _databasePath);
        }
    }

    public void Upsert(string tableName, string documentId, TDocument document)
    {
        var handle = GetColumnFamilyHandle(tableName);
        var keyBytes = Encoding.UTF8.GetBytes(documentId);
        var valueBytes = JsonSerializer.SerializeToUtf8Bytes(document, _jsonTypeInfo);

        unsafe
        {
            fixed (byte* keyPtr = keyBytes)
            fixed (byte* valuePtr = valueBytes)
            {
                RocksDbNativeMethods.PutCf(
                    _dbHandle, _writeOptions, handle,
                    (nint)keyPtr, (nuint)keyBytes.Length,
                    (nint)valuePtr, (nuint)valueBytes.Length,
                    out var errPtr);
                ThrowOnError(errPtr, $"Failed to put key '{documentId}' in table '{tableName}'");
            }
        }
    }

    public void Delete(string tableName, string documentId)
    {
        var handle = GetColumnFamilyHandle(tableName);
        var keyBytes = Encoding.UTF8.GetBytes(documentId);

        unsafe
        {
            fixed (byte* keyPtr = keyBytes)
            {
                RocksDbNativeMethods.DeleteCf(
                    _dbHandle, _writeOptions, handle,
                    (nint)keyPtr, (nuint)keyBytes.Length,
                    out var errPtr);
                ThrowOnError(errPtr, $"Failed to delete key '{documentId}' from table '{tableName}'");
            }
        }
    }

    public TDocument? Read(string tableName, string documentId)
    {
        var handle = GetColumnFamilyHandle(tableName);
        var keyBytes = Encoding.UTF8.GetBytes(documentId);

        nint resultPtr;
        nuint resultLength;

        unsafe
        {
            fixed (byte* keyPtr = keyBytes)
            {
                resultPtr = RocksDbNativeMethods.GetCf(
                    _dbHandle, _readOptions, handle,
                    (nint)keyPtr, (nuint)keyBytes.Length,
                    out resultLength, out var errPtr);
                ThrowOnError(errPtr, $"Failed to get key '{documentId}' from table '{tableName}'");
            }
        }

        if (resultPtr == nint.Zero)
            return null;

        try
        {
            unsafe
            {
                var span = new ReadOnlySpan<byte>((void*)resultPtr, (int)resultLength);
                return JsonSerializer.Deserialize(span, _jsonTypeInfo);
            }
        }
        finally
        {
            RocksDbNativeMethods.Free(resultPtr);
        }
    }

    public bool Exists(string tableName, string documentId)
    {
        return Read(tableName, documentId) is not null;
    }

    public IReadOnlyList<string> ListKeys(string tableName)
    {
        var handle = GetColumnFamilyHandle(tableName);
        var keys = new List<string>();

        var iterator = RocksDbNativeMethods.CreateIteratorCf(_dbHandle, _readOptions, handle);
        try
        {
            RocksDbNativeMethods.IterSeekToFirst(iterator);

            while (RocksDbNativeMethods.IterValid(iterator) != 0)
            {
                var keyPtr = RocksDbNativeMethods.IterKey(iterator, out var keyLength);

                unsafe
                {
                    var keySpan = new ReadOnlySpan<byte>((void*)keyPtr, (int)keyLength);
                    keys.Add(Encoding.UTF8.GetString(keySpan));
                }

                RocksDbNativeMethods.IterNext(iterator);
            }
        }
        finally
        {
            RocksDbNativeMethods.IterDestroy(iterator);
        }

        return keys;
    }

    private nint GetColumnFamilyHandle(string tableName)
    {
        if (_columnFamilyHandles.TryGetValue(tableName, out var handle))
            return handle;

        throw new InvalidOperationException(
            $"Table '{tableName}' has not been initialized. Call InitializeTable first.");
    }

    /// <summary>
    /// Allocates a null-terminated UTF-8 string in unmanaged memory. Caller must free with Marshal.FreeHGlobal.
    /// </summary>
    private static nint MarshalUtf8String(string value)
    {
        var utf8Bytes = Encoding.UTF8.GetBytes(value);
        var ptr = Marshal.AllocHGlobal(utf8Bytes.Length + 1);
        Marshal.Copy(utf8Bytes, 0, ptr, utf8Bytes.Length);
        Marshal.WriteByte(ptr, utf8Bytes.Length, 0); // null terminator
        return ptr;
    }

    /// <summary>
    /// Check an error pointer returned by a RocksDB C API call. If non-null, reads the error
    /// message, frees the native string, and throws InternalErrorException.
    /// </summary>
    private static void ThrowOnError(nint errPtr, string context)
    {
        if (errPtr == nint.Zero)
            return;

        var message = Marshal.PtrToStringUTF8(errPtr) ?? "Unknown RocksDB error";
        RocksDbNativeMethods.Free(errPtr);
        throw new InternalErrorException($"{context}: {message}");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_writeOptions != nint.Zero)
        {
            RocksDbNativeMethods.WriteOptionsDestroy(_writeOptions);
            _writeOptions = nint.Zero;
        }

        if (_readOptions != nint.Zero)
        {
            RocksDbNativeMethods.ReadOptionsDestroy(_readOptions);
            _readOptions = nint.Zero;
        }

        foreach (var kvp in _columnFamilyHandles)
        {
            RocksDbNativeMethods.ColumnFamilyHandleDestroy(kvp.Value);
        }
        _columnFamilyHandles.Clear();

        if (_dbHandle != nint.Zero)
        {
            RocksDbNativeMethods.Close(_dbHandle);
            _dbHandle = nint.Zero;
        }

        GC.SuppressFinalize(this);
    }
}