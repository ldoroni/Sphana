using System.Runtime.InteropServices;

namespace Sphana.DataStore.Infrastructure.Persistence.RocksDb;

/// <summary>
/// P/Invoke declarations for the RocksDB C library.
/// Requires native librocksdb.so (Linux) or rocksdb.dll (Windows) on the library path.
/// All functions use the stable rocksdb C API (rocksdb/c.h).
/// </summary>
internal static partial class RocksDbNativeMethods
{
    private const string RocksDbLibrary = "rocksdb";

    // ── Options ────────────────────────────────────────────────────────────
    // rocksdb_options_t is used for both DB-level and per-column-family options
    // in rocksdb_open_column_families.

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_options_create")]
    internal static partial nint OptionsCreate();

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_options_destroy")]
    internal static partial void OptionsDestroy(nint options);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_options_set_create_if_missing")]
    internal static partial void OptionsSetCreateIfMissing(nint options, byte value);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_options_set_create_missing_column_families")]
    internal static partial void OptionsSetCreateMissingColumnFamilies(nint options, byte value);

    // ── Read / Write Options ───────────────────────────────────────────────

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_readoptions_create")]
    internal static partial nint ReadOptionsCreate();

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_readoptions_destroy")]
    internal static partial void ReadOptionsDestroy(nint readOptions);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_writeoptions_create")]
    internal static partial nint WriteOptionsCreate();

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_writeoptions_destroy")]
    internal static partial void WriteOptionsDestroy(nint writeOptions);

    // ── Database Lifecycle ─────────────────────────────────────────────────

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_open", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint Open(nint options, string path, out nint errPtr);

    /// <summary>
    /// Open a database with column families.
    /// column_family_names: array of nint, each pointing to a null-terminated UTF-8 string.
    /// column_family_options: array of nint, each a rocksdb_options_t*.
    /// column_family_handles: output array of nint, each a rocksdb_column_family_handle_t*.
    /// </summary>
    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_open_column_families", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint OpenColumnFamilies(
        nint options,
        string path,
        int numColumnFamilies,
        nint[] columnFamilyNames,
        nint[] columnFamilyOptions,
        nint[] columnFamilyHandles,
        out nint errPtr);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_close")]
    internal static partial void Close(nint db);

    // ── Column Family Management ───────────────────────────────────────────

    /// <summary>
    /// Returns a char** (array of C strings). Caller must read count pointers from the returned
    /// address, then free via ListColumnFamiliesDestroy.
    /// </summary>
    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_list_column_families", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint ListColumnFamilies(
        nint options,
        string path,
        out nuint count,
        out nint errPtr);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_list_column_families_destroy")]
    internal static partial void ListColumnFamiliesDestroy(nint list, nuint count);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_create_column_family", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint CreateColumnFamily(
        nint db,
        nint columnFamilyOptions,
        string columnFamilyName,
        out nint errPtr);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_drop_column_family")]
    internal static partial void DropColumnFamily(
        nint db,
        nint columnFamilyHandle,
        out nint errPtr);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_column_family_handle_destroy")]
    internal static partial void ColumnFamilyHandleDestroy(nint columnFamilyHandle);

    // ── Put / Get / Delete (Column Family) ─────────────────────────────────

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_put_cf")]
    internal static partial void PutCf(
        nint db,
        nint writeOptions,
        nint columnFamilyHandle,
        nint key,
        nuint keyLength,
        nint value,
        nuint valueLength,
        out nint errPtr);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_get_cf")]
    internal static partial nint GetCf(
        nint db,
        nint readOptions,
        nint columnFamilyHandle,
        nint key,
        nuint keyLength,
        out nuint valueLength,
        out nint errPtr);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_delete_cf")]
    internal static partial void DeleteCf(
        nint db,
        nint writeOptions,
        nint columnFamilyHandle,
        nint key,
        nuint keyLength,
        out nint errPtr);

    // ── Iterator (Column Family) ───────────────────────────────────────────

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_create_iterator_cf")]
    internal static partial nint CreateIteratorCf(
        nint db,
        nint readOptions,
        nint columnFamilyHandle);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_iter_seek_to_first")]
    internal static partial void IterSeekToFirst(nint iterator);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_iter_valid")]
    internal static partial byte IterValid(nint iterator);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_iter_next")]
    internal static partial void IterNext(nint iterator);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_iter_key")]
    internal static partial nint IterKey(nint iterator, out nuint keyLength);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_iter_value")]
    internal static partial nint IterValue(nint iterator, out nuint valueLength);

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_iter_destroy")]
    internal static partial void IterDestroy(nint iterator);

    // ── Memory ─────────────────────────────────────────────────────────────

    [LibraryImport(RocksDbLibrary, EntryPoint = "rocksdb_free")]
    internal static partial void Free(nint ptr);
}