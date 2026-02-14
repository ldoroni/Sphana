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
| **Context Manager** | Locks and cache lifecycle support `with` statement |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       DistributedCache                           │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐   │
│  │  CacheCollection  │  │  CacheCollection  │  │  Global Locks │   │
│  │  "sessions"       │  │  "users"          │  │  .lock()      │   │
│  │  .put/.get/.del   │  │  .put/.get/.del   │  │  .try_lock()  │   │
│  └────────┬──────────┘  └────────┬──────────┘  └──────┬────────┘   │
│           │                      │                     │           │
│  ┌────────▼──────────────────────▼─────────────────────▼────────┐ │
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