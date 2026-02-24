using System.Text.Json.Serialization;
using Sphana.DataStore.Domain.Models;
using Sphana.DataStore.Presentation.Schemas.Embeddings;
using Sphana.DataStore.Presentation.Schemas.Entries;
using Sphana.DataStore.Presentation.Schemas.Indices;
using Sphana.DataStore.Presentation.Schemas.Payloads;
using Sphana.DataStore.Presentation.Schemas.Queries;
using Sphana.RequestHandler;

namespace Sphana.DataStore.Infrastructure.Serialization;

/// <summary>
/// AOT-compatible JSON serializer context using source generation for all serialized types.
/// </summary>
// ── Presentation Schemas: Indices ──────────────────────────────────────────
[JsonSerializable(typeof(CreateIndexRequest))]
[JsonSerializable(typeof(DeleteIndexRequest))]
[JsonSerializable(typeof(GetIndexRequest))]
[JsonSerializable(typeof(IndexDetailsResponse))]
[JsonSerializable(typeof(IndexExistsRequest))]
[JsonSerializable(typeof(IndexExistsResponse))]
[JsonSerializable(typeof(UpdateIndexRequest))]
// ── Presentation Schemas: Embeddings ───────────────────────────────────────
[JsonSerializable(typeof(AddEmbeddingsRequest))]
[JsonSerializable(typeof(AddEmbeddingsEntrySchema))]
[JsonSerializable(typeof(AddEmbeddingsResponse))]
[JsonSerializable(typeof(ResetEmbeddingsRequest))]
[JsonSerializable(typeof(ResetEmbeddingsResponse))]
// ── Presentation Schemas: Entries ──────────────────────────────────────────
[JsonSerializable(typeof(DeleteEntryRequest))]
[JsonSerializable(typeof(ListEntriesRequest))]
[JsonSerializable(typeof(ListEntriesResponse))]
// ── Presentation Schemas: Payloads ─────────────────────────────────────────
[JsonSerializable(typeof(UploadPayloadRequest))]
[JsonSerializable(typeof(AppendPayloadRequest))]
[JsonSerializable(typeof(DeletePayloadRequest))]
[JsonSerializable(typeof(PayloadResponse))]
// ── Presentation Schemas: Queries ──────────────────────────────────────────
[JsonSerializable(typeof(ExecuteQueryRequest))]
[JsonSerializable(typeof(ExecuteQueryResultSchema))]
// ── Domain Models (used by RocksDB repository serialization) ───────────────
[JsonSerializable(typeof(IndexDetails))]
[JsonSerializable(typeof(ShardDetails))]
[JsonSerializable(typeof(EmbeddingDetails))]
[JsonSerializable(typeof(EmbeddingResult))]
[JsonSerializable(typeof(ExecuteQueryResult))]
// ── Error handling (used by ManagedExceptionMiddleware) ────────────────────
[JsonSerializable(typeof(ErrorResponse))]
// ── Common collection types ────────────────────────────────────────────────
[JsonSerializable(typeof(List<IndexDetailsResponse>))]
[JsonSerializable(typeof(List<string>))]
[JsonSerializable(typeof(List<ExecuteQueryResultSchema>))]
[JsonSerializable(typeof(List<AddEmbeddingsEntrySchema>))]
[JsonSerializable(typeof(Dictionary<string, string>))]
public partial class AppJsonSerializerContext : JsonSerializerContext;