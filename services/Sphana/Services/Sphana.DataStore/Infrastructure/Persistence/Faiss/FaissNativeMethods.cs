using System.Runtime.InteropServices;

namespace Sphana.DataStore.Infrastructure.Persistence.Faiss;

/// <summary>
/// P/Invoke declarations for the FAISS C library (libfaiss_c).
/// Requires native libfaiss_c.so (Linux) or faiss_c.dll (Windows) on the library path.
/// </summary>
internal static partial class FaissNativeMethods
{
    private const string FaissLibrary = "faiss_c";

    [LibraryImport(FaissLibrary, EntryPoint = "faiss_IndexFlatL2_new_with")]
    internal static partial int IndexFlatL2NewWith(out nint indexPointer, int dimension);

    [LibraryImport(FaissLibrary, EntryPoint = "faiss_Index_add")]
    internal static partial int IndexAdd(nint indexPointer, long numberOfVectors, float[] vectors);

    [LibraryImport(FaissLibrary, EntryPoint = "faiss_Index_search")]
    internal static partial int IndexSearch(
        nint indexPointer,
        long numberOfQueries,
        float[] queryVectors,
        long numberOfResults,
        float[] distances,
        long[] labels);

    [LibraryImport(FaissLibrary, EntryPoint = "faiss_Index_reset")]
    internal static partial int IndexReset(nint indexPointer);

    [LibraryImport(FaissLibrary, EntryPoint = "faiss_Index_ntotal")]
    internal static partial long IndexNTotal(nint indexPointer);

    [LibraryImport(FaissLibrary, EntryPoint = "faiss_Index_free")]
    internal static partial int IndexFree(nint indexPointer);

    [LibraryImport(FaissLibrary, EntryPoint = "faiss_write_index_fname", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int WriteIndexToFile(nint indexPointer, string filePath);

    [LibraryImport(FaissLibrary, EntryPoint = "faiss_read_index_fname", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int ReadIndexFromFile(string filePath, int ioFlags, out nint indexPointer);
}