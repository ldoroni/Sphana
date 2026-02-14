# Get Started

## Install Locally
Run the following commands:
1. Install Python Dependencies:<br>
   `uv sync --project .\libraries\distributed-cache-lib`
2. Activate the Virtual Environment:<br>
   `.\libraries\distributed-cache-lib\.venv\Scripts\activate`

## Build Library
Run the following command:
1. Build library:<br>
   `uv build --project .\libraries\distributed-cache-lib`
2. Publish to repository:<br>
   `uv publish --publish-url http://localhost:61000/  .\libraries\distributed-cache-lib\dist\*`

----------

# Distributed Cache Library

A production-ready, embedded distributed cache for Python applications running across multiple Kubernetes pods or any networked nodes. Uses a **Hazelcast-style partitioned architecture** — no external dependencies like Redis or etcd required.

## Features

| Feature | Description |
|---|---|
| **Partitioned Storage** | Data distributed across 271 virtual partitions for linear scalability |
| **Collections** | Named key-value namespaces with independent TTL and capacity settings |
| **TTL Expiration** | Per-entry and per-collection default time-to-live |
| **LRU Eviction** | Max capacity per collection with least-recently-used eviction |
| **Distributed Locking** | Per-key locks with fencing tokens, acquire & hold timeouts |
| **Backup Replication** | Synchronous in-memory replication to backup nodes |
| **WAL Persistence** | Write-Ahead Log for crash-recovery disk persistence |
| **Quorum Protection** | Writes rejected during network partitions to prevent split-brain |
| **Anti-Entropy** | Background repair of backup drift via periodic snapshots |
| **Dynamic Membership** | Pods can join/leave at any time; cluster auto-rebalances |
| **Batch Operations** | `put_many`, `get_many`, `delete_many` for bulk operations |
| **Task Routing** | Route arbitrary tasks to partition owners with automatic failover |
| **Context Manager** | Locks and cache lifecycle support `with` statement |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       DistributedCache                           │
│                                                                  │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────┐ ┌──────────┐│
│  │CacheCollection │ │CacheCollection │ │Global Locks│ │TaskRouter││
│  │ "sessions"     │ │ "users"        │ │ .lock()    │ │ .submit()││
│  │ .put/.get/.del │ │ .put/.get/.del │ │ .try_lock()│ │ .listen()││
│  └───────┬────────┘ └───────┬────────┘ └─────┬──────┘ └────┬─────┘│
│           │                      │                     │        │  │
│  ┌────────▼──────────────────────▼─────────────────────▼────────▼┐ │
│  │                    Partitioning Layer                         │ │
│  │  ┌─────────────────┐ ┌────────────────┐ ┌─────────────────┐  │ │
│  │  │PartitionStrategy│ │ PartitionTable │ │PartitionMigrator│  │ │
│  │  │ key → partition  │ │ part → node    │ │ data transfer   │  │ │
│  │  └─────────────────┘ └────────────────┘ └─────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                      Storage Layer                           │ │
│  │  ┌───────────────┐ ┌─────────────────┐ ┌──────────────────┐ │ │
│  │  │PartitionStore │ │BackupReplicator │ │    WalStore      │ │ │
│  │  │ in-memory data│ │ sync to backups │ │ disk persistence │ │ │
│  │  │ + LRU eviction│ │ + anti-entropy  │ │ + compaction     │ │ │
│  │  └───────────────┘ └─────────────────┘ └──────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                      Cluster Layer                           │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │ │
│  │  │  Coordinator  │ │ NodeRegistry │ │ClusterNodesProvider  │ │ │
│  │  │oldest-member  │ │  heartbeats  │ │  (user-provided)     │ │ │
│  │  │ election      │ │  15s timeout │ │  pod IP discovery    │ │ │
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                      Network Layer                           │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │ │
│  │  │  RpcServer    │ │  RpcClient   │ │  ConnectionPool     │ │ │
│  │  │  TCP listener │ │  TCP sender  │ │  4 conns per peer   │ │ │
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
          ↕ TCP (msgpack RPC)              ↕ TCP (msgpack RPC)
    ┌─────────────────┐              ┌─────────────────┐
    │  Pod 2 (Node B)  │              │  Pod 3 (Node C)  │
    │  owns partitions │              │  owns partitions │
    │  90-180          │              │  181-271         │
    └─────────────────┘              └─────────────────┘
```

### Data Flow

```
  Client: cache.put("user:123", data)
    │
    ▼
  PartitionStrategy: hash("user:123") → partition #42
    │
    ▼
  PartitionTable: partition #42 → primary: Node B, backup: Node C
    │
    ├──── Local? ──► PartitionStore.put() → WalStore.append()
    │                                         │
    │                                         ▼
    │                               BackupReplicator → RPC to Node C
    │
    └──── Remote? ──► RpcClient.send(PUT, key, value) → Node B handles locally
```

### Locking Flow

```
  Client: cache.acquire_lock("resource-x")
    │
    ▼
  PartitionStrategy: hash("resource-x") → partition #87
    │
    ▼
  PartitionTable: partition #87 → owner: Node C
    │
    ├──── Local? ──► LockManager.acquire() → fencing_token++
    │
    └──── Remote? ──► RPC LOCK_ACQUIRE to Node C
                        │
                        ▼
                    Node C: LockManager.acquire() → return fencing_token
```

### Key Components

| Component | Description |
|---|---|
| `DistributedCache` | Main entry point — orchestrates all subsystems |
| `CacheCollection` | Named key-value collection with TTL support |
| `ClusterNodesProvider` | Abstract class — supplies current cluster member addresses |
| `NodeRegistry` | Tracks cluster members via heartbeats |
| `Coordinator` | Elected leader that manages partition assignments |
| `PartitionStrategy` | Maps keys → partition IDs via consistent hashing |
| `PartitionTable` | Tracks which node owns which partition (primary + backups) |
| `PartitionRebalancer` | Computes new partition assignments on membership changes |
| `PartitionMigrator` | Transfers partition data between nodes |
| `BackupReplicator` | Synchronous in-memory replication to backup nodes |
| `WalStore` | Write-Ahead Log for crash-recovery disk persistence |
| `LockManager` | Distributed locking with fencing tokens |
| `DistributedTaskRouter` | Routes arbitrary tasks to partition owners with retry and failover |
| `TaskResult` | Result model for task routing (success/failure, response, error, node address) |
| `RpcServer` / `RpcClient` | TCP-based RPC with msgpack serialization |
| `ConnectionPool` | Reusable TCP connection pool to peers |

### Network Protocol

- **TCP sockets** with length-prefixed msgpack frames
- **Connection pooling** (4 connections per peer by default)
- **Heartbeat-based failure detection** (15s timeout)

## Quick Start

```python
from distributed_cache import DistributedCache, ClusterNodesProvider

class K8sProvider(ClusterNodesProvider):
    def get_node_addresses(self) -> list[str]:
        # Return current pod IP:port list from Kubernetes API
        return ["10.0.0.1:9100", "10.0.0.2:9100", "10.0.0.3:9100"]

# Create and start
cache = DistributedCache(
    self_address="10.0.0.1:9100",
    cluster_nodes_provider=K8sProvider(),
)
cache.start()

# Basic cache operations
cache.put("user:123", {"name": "Alice", "role": "admin"}, ttl=300)
user = cache.get("user:123")
cache.delete("user:123")

# Named collections
sessions = cache.get_collection("sessions", default_ttl=3600)
sessions.put("sess:abc", {"user_id": 123})
sessions.get("sess:abc")

# Distributed locking
with cache.acquire_lock("resource-x", hold_timeout=30):
    # Critical section — only one node can execute this at a time
    pass

# Task routing (route operations to partition owners)
router = cache.get_task_router()
router.listen("shard.ingest", lambda routing_key, msg: {"status": "ok"})
result = router.submit("shard-0", "shard.ingest", {"doc": "..."})

# Non-blocking lock attempt
lock = cache.try_acquire_lock("resource-y")
if lock:
    try:
        # Got the lock
        pass
    finally:
        lock.release()

# Context manager for lifecycle
with DistributedCache(
    self_address="10.0.0.1:9100",
    cluster_nodes_provider=K8sProvider(),
) as cache:
    cache.put("key", "value")

cache.stop()
```

## Configuration

```python
cache = DistributedCache(
    self_address="10.0.0.1:9100",
    cluster_nodes_provider=provider,
    partition_count=271,             # Number of hash-ring partitions
    backup_count=1,                  # Backup replicas per partition (0 = disabled)
    rpc_port=9100,                   # TCP port for RPC (0 = extract from address)
    connection_timeout=5.0,          # TCP connect timeout (seconds)
    request_timeout=10.0,            # RPC request timeout (seconds)
    wal_dir="/data/wal",             # WAL directory (None = disabled, pure in-memory)
    wal_fsync=True,                  # fsync after every WAL write
    wal_max_segment_bytes=67108864,  # Max WAL segment size before rotation (64 MiB)
)
```

## Data Durability

The cache supports two layers of data protection that can be used independently or together:

### Layer 1: In-Memory Backup Replication

When `backup_count >= 1`, every write (put/delete/clear) on the primary partition owner is **synchronously replicated** to backup nodes via RPC. If the primary node crashes, the backup node already holds the data in memory and can be promoted during rebalancing with zero data loss.

- Backup failures are **logged but do not block** the primary write
- On rebalance, newly-assigned backup nodes receive a **full partition snapshot**
- Works without any disk — pure in-memory redundancy

```python
# Enable 1 backup replica per partition (default)
cache = DistributedCache(..., backup_count=1)

# Disable backups (not recommended for production)
cache = DistributedCache(..., backup_count=0)
```

### Layer 2: Write-Ahead Log (WAL) Persistence

When `wal_dir` is set, every cache mutation is appended to an on-disk WAL **before** the in-memory write completes. On crash-recovery, the WAL is replayed to restore in-memory state.

- **Segmented files**: WAL rotates to new segment when current exceeds `wal_max_segment_bytes`
- **Background compaction**: Old, fully-replayed segments are periodically removed (every 5 min)
- **fsync control**: Set `wal_fsync=False` for higher throughput at the cost of durability on power loss
- **Binary format**: Uses msgpack-encoded records with length-prefix framing

```python
# Enable WAL persistence
cache = DistributedCache(..., wal_dir="/data/wal")

# High-throughput mode (skip fsync, rely on OS buffer flush)
cache = DistributedCache(..., wal_dir="/data/wal", wal_fsync=False)

# Pure in-memory mode (no WAL)
cache = DistributedCache(..., wal_dir=None)  # default
```

### Combining Both Layers

For maximum durability, use both backup replication and WAL:

```python
cache = DistributedCache(
    self_address="10.0.0.1:9100",
    cluster_nodes_provider=provider,
    backup_count=1,           # In-memory backup on another node
    wal_dir="/data/wal",      # Disk persistence on this node
)
```

This gives you:
- **Node crash**: Backup node has all data in memory → instant promotion
- **Full cluster restart**: WAL replay restores state from disk
- **Network partition**: WAL ensures local durability even when backups are unreachable

## Collection Management

Collections are named key-value namespaces with their own TTL and capacity settings:

```python
# Create / get a collection with defaults
sessions = cache.get_collection("sessions", default_ttl=3600, max_entries=10000)

# Bulk operations
sessions.put_many({"key1": "val1", "key2": "val2"}, ttl=60)
results = sessions.get_many(["key1", "key2"])
deleted = sessions.delete_many(["key1", "key2"])

# Introspection
all_keys = sessions.keys()
count = sessions.size()
sessions.clear()

# Collection lifecycle
cache.list_collections()          # → ["sessions", "__default__"]
cache.collection_exists("sessions")  # → True
cache.delete_collection("sessions")  # clears all entries and removes
```

### LRU Eviction

When `max_entries` is set on a collection, the least-recently-used entries are automatically evicted when the limit is exceeded:

```python
lru_cache = cache.get_collection("hot-data", max_entries=1000, default_ttl=300)
# When the 1001st entry is inserted, the oldest-accessed entry is evicted
```

## Quorum Protection

Write operations (`put`, `delete`, `clear`, `put_many`, `delete_many`) are rejected with `QuorumLostError` when the cluster cannot see a majority of its members. This prevents split-brain data corruption:

```python
from distributed_cache import QuorumLostError

try:
    cache.put("key", "value")
except QuorumLostError:
    # Cluster is partitioned — wait for recovery
    pass
```

## Read-from-Backup Fallback

When a `get()` request fails because the primary partition owner is unreachable, the library automatically falls back to reading from backup nodes. This provides read availability even during node failures or network partitions.

## Anti-Entropy

A background thread periodically pushes full partition snapshots from primary owners to their backup nodes, repairing any drift caused by missed replication messages. This runs every 60 seconds when `backup_count >= 1`.

## Distributed Task Routing

The `DistributedTaskRouter` extends the partition-based architecture beyond cache data — it lets you route **arbitrary tasks** (function calls, operations, commands) to the node that owns a given partition, based on a routing key.

This is the mechanism that makes write operations (document ingest, index creation, etc.) fault-tolerant: instead of directly calling a specific pod, you submit a task with a routing key, and the library handles node resolution, local-vs-remote dispatch, retries, and automatic failover when nodes die.

### Core Concepts

| Concept | Description |
|---|---|
| **Routing key** | A string (e.g. shard name) hashed to determine which node owns the task |
| **Topic** | A named operation type (e.g. `"shard.ingest_document"`) — like a message topic |
| **Listener** | A local handler function registered for a topic — invoked when the task lands on this node |
| **TaskResult** | The result of a task submission — contains status, response, error, and node address |

### Getting the Task Router

```python
router = cache.get_task_router()
```

The router is a singleton per `DistributedCache` instance. It uses the same partition strategy and partition table as the cache itself.

### Registering Listeners

Register a handler for a topic on the current node. The handler receives the routing key and the message dict:

```python
def handle_ingest(routing_key: str, message: dict) -> dict | None:
    shard_name = routing_key
    document = message["document"]
    # ... perform the ingest operation locally ...
    return {"document_id": "abc123"}

router.listen("shard.ingest_document", handle_ingest)
```

Every node in the cluster should register the same set of listeners during startup. When a task is routed to a node, the listener for the matching topic is invoked locally.

### Submitting Tasks (Single Target)

Submit a task to the node that owns the partition for the given routing key:

```python
from distributed_cache import TaskResult

result: TaskResult = router.submit(
    routing_key="shard-0",                # Determines which node handles this
    topic="shard.ingest_document",        # Which listener to invoke
    message={"document": {...}},          # Arbitrary payload dict
    timeout=120.0,                        # RPC timeout (seconds)
    max_retries=2,                        # Retry on connection/timeout errors
)

if result.is_success:
    print(result.response)   # The dict returned by the listener
else:
    print(result.error)      # Error message string
    print(result.node_address)  # Which node failed
```

### How Routing Works

```
  router.submit("shard-0", "shard.ingest_document", {...})
    │
    ▼
  PartitionStrategy: hash("shard-0") → partition #42
    │
    ▼
  PartitionTable: partition #42 → owner: Node B
    │
    ├──── Node B is this node? ──► Invoke listener directly
    │                                 └──► Return TaskResult.success(response)
    │
    └──── Node B is remote? ──► RPC TASK_SUBMIT to Node B
                                    │
                                    ├──── Success ──► Return TaskResult.success(response)
                                    │
                                    └──── Failure ──► Retry with backoff
                                                        └──► Return TaskResult.failure(error)
```

### Broadcasting to All Nodes

For operations that must execute on **every** node (e.g. `create_index`, `delete_index`), use `submit_to_all`:

```python
results: dict[str, TaskResult] = router.submit_to_all(
    topic="shard.create_index",
    message={"index_name": "my-index", "config": {...}},
    timeout=120.0,
)

for node_address, result in results.items():
    if result.is_success:
        print(f"{node_address}: OK")
    else:
        print(f"{node_address}: FAILED — {result.error}")
```

`submit_to_all` iterates all unique node addresses in the partition table and executes the task on each one (locally if this node, via RPC if remote).

### TaskResult

`TaskResult` is a Pydantic model returned by `submit()` and `submit_to_all()`:

```python
from distributed_cache import TaskResult, TaskStatus

# Fields
result.status         # TaskStatus.SUCCESS or TaskStatus.FAILURE
result.response       # dict | None — the handler's return value
result.error          # str | None — error message on failure
result.node_address   # str | None — which node executed (or failed)

# Convenience properties
result.is_success     # True if status == SUCCESS
result.is_failure     # True if status == FAILURE

# Factory methods (used internally)
TaskResult.success(response={"id": "123"}, node_address="10.0.0.2:9100")
TaskResult.failure(error="Connection refused", node_address="10.0.0.2:9100")
```

### Fault Tolerance

The task router provides automatic fault tolerance through several mechanisms:

1. **Retry with exponential backoff**: When `max_retries > 0`, transient RPC failures (connection errors, timeouts) trigger retries with increasing delay. Logical errors (the handler raised an exception) are **not** retried.

2. **Automatic rerouting after node failure**: When a node dies, the coordinator detects the missing heartbeat, the rebalancer reassigns its partitions to surviving nodes, and the partition table is updated cluster-wide. Subsequent `submit()` calls with the same routing key are automatically routed to the new owner — no caller-side logic needed.

3. **Local fast-path**: When the routing key maps to the current node, the listener is invoked directly in-process — no network overhead.

### Listener Handler Signature

```python
def handler(routing_key: str, message: dict) -> dict | None:
    """
    Args:
        routing_key: The routing key from the submit() call (e.g. shard name).
        message: The message dict from the submit() call.

    Returns:
        A dict that becomes TaskResult.response, or None.

    Raises:
        Any exception — caught and returned as TaskResult.failure(error=str(exc)).
    """
```

### Example: Shard-Routed Write Operations

```python
# ── Startup (every pod) ──────────────────────────────
router = cache.get_task_router()

def handle_ingest(routing_key: str, message: dict) -> dict:
    shard_name = routing_key
    doc_data = message["document"]
    # ... write to local shard storage ...
    return {"document_id": "doc-123", "shard": shard_name}

def handle_delete(routing_key: str, message: dict) -> dict:
    shard_name = routing_key
    doc_id = message["document_id"]
    # ... delete from local shard storage ...
    return {"deleted": True}

router.listen("shard.ingest_document", handle_ingest)
router.listen("shard.delete_document", handle_delete)

# ── Request handling (any pod) ────────────────────────
# The request can arrive at any pod — it will be routed to the correct one
result = router.submit("shard-0", "shard.ingest_document", {
    "document": {"title": "Hello", "content": "..."}
})

if result.is_success:
    return result.response  # {"document_id": "doc-123", "shard": "shard-0"}
else:
    raise Exception(f"Ingest failed on {result.node_address}: {result.error}")
```

## Distributed Locking

Locks are partitioned like cache entries — the lock key determines which node is responsible:

- **Fencing tokens**: Monotonically increasing tokens prevent stale lock holders from corrupting state
- **Hold timeout**: Auto-release after a configurable duration (default 60s)
- **Acquire timeout**: Blocking wait with configurable timeout (default 30s)
- **Force release**: Admin operation to release stuck locks

```python
# Blocking acquire with timeouts
lock = cache.acquire_lock(
    "my-resource",
    hold_timeout=60.0,       # Max time to hold the lock
    acquire_timeout=30.0,    # Max time to wait for acquisition
    retry_interval=0.2,      # Sleep between retries
)

# Use as context manager
with cache.acquire_lock("my-resource") as lock:
    print(f"Fencing token: {lock.fencing_token}")
```

## Exceptions

| Exception | When |
|---|---|
| `LockAcquireTimeoutError` | Lock not acquired within `acquire_timeout` |
| `LockHoldTimeoutError` | Lock expired while held (raised on `release()`) |
| `LockNotHeldError` | Attempted to release a lock not owned by this handle |
| `QuorumLostError` | Write attempted during network partition (no majority) |
| `ClusterNotReadyError` | Operation attempted before cluster has formed |
| `CollectionNotFoundError` | Operation on a non-existent collection |
| `PartitionNotOwnedError` | Request routed to wrong node (stale partition table) |

## Module Structure

```
distributed_cache/
├── __init__.py                     # Public API exports
├── distributed_cache.py            # Main orchestrator
├── cache_collection.py             # Per-collection facade
├── models.py                       # Data models (RPC message types, etc.)
├── exceptions.py                   # Exception hierarchy
├── routing/
│   ├── __init__.py                 # Routing module exports
│   ├── task_router.py              # DistributedTaskRouter — partition-based task dispatch
│   └── task_result.py              # TaskResult / TaskStatus models
├── cluster/
│   ├── cluster_nodes_provider.py   # Abstract node discovery interface
│   ├── coordinator.py              # Oldest-member coordinator election
│   └── node_registry.py            # Heartbeat-based membership tracking
├── partitioning/
│   ├── partition_strategy.py       # Key → partition hashing
│   ├── partition_table.py          # Partition → node assignment map
│   ├── partition_rebalancer.py     # Rebalance computation on membership change
│   └── partition_migrator.py       # Data transfer during rebalancing
├── storage/
│   ├── partition_store.py          # In-memory partition data with LRU
│   ├── backup_replicator.py        # Sync replication + anti-entropy
│   ├── lock_store.py               # Per-partition lock state
│   └── wal_store.py                # Write-Ahead Log persistence
├── locking/
│   ├── lock_manager.py             # Distributed lock coordination
│   └── lock_handle.py              # Lock context manager
└── network/
    ├── rpc_protocol.py             # Message framing (length-prefix + msgpack)
    ├── rpc_server.py               # TCP server for incoming RPC
    ├── rpc_client.py               # TCP client for outgoing RPC
    └── connection_pool.py          # Reusable connection pool per peer
```

## Dependencies

- **msgpack** — Binary serialization for RPC messages and WAL records
- **pydantic** — Data model validation
- **Python 3.12** standard library (threading, socket, struct, os)

No external databases, no file-system locks — everything is embedded and communicates over TCP.

## Requirements

- Python 3.12+
- `msgpack >= 1.0`
- `pydantic >= 2.0`
- TCP connectivity between pods on the RPC port (default 9100)