# RocksDB Native Library Compiler

Builds RocksDB shared libraries (`librocksdb.so` for Linux, `rocksdb.dll` for Windows) from source using Docker.

These native libraries are used by the `Sphana.DataStore` .NET service via P/Invoke (`RocksDbNativeMethods.cs`), replacing the `RocksDbSharp` NuGet package to support .NET AOT compilation.

## Prerequisites

- Docker Desktop

## Quick Start

```powershell
.\utils\rocksdb-compiler-util\run.ps1 -RuntimesDir "..\..\services\Sphana\Services\Sphana.DataStore\runtimes" 
```

### More Options

```powershell
# Build both platforms (output to ./target)
.\utils\rocksdb-compiler-util\run.ps1

# Build only Linux
.\utils\rocksdb-compiler-util\run.ps1 -Target linux

# Build only Windows
.\utils\rocksdb-compiler-util\run.ps1 -Target windows

# Custom output directory
.\utils\rocksdb-compiler-util\run.ps1 -RuntimesDir "..\..\services\Sphana\Services\Sphana.DataStore\runtimes"
```

## Parameters

| Parameter      | Default     | Description                                      |
|----------------|-------------|--------------------------------------------------|
| `-Target`      | `all`       | Build target: `all`, `linux`, or `windows`       |
| `-RuntimesDir` | `./target`  | Output directory for the runtimes folder layout  |

## Output

Libraries are copied to the runtimes directory in the standard .NET native library layout:

```
<RuntimesDir>/
├── linux-x64/native/
│   └── librocksdb.so
└── win-x64/native/
    └── rocksdb.dll
```

## Deployment

The `RocksDbNativeMethods.cs` P/Invoke layer expects the library name `rocksdb` and will resolve to `rocksdb.dll` on Windows or `librocksdb.so` on Linux automatically.

To deploy to the `Sphana.DataStore` service:

```powershell
.\utils\rocksdb-compiler-util\run.ps1 -RuntimesDir "..\..\services\Sphana\Services\Sphana.DataStore\runtimes"
```

## Dockerfiles

| Dockerfile          | Platform    | Base Image                    | Container Mode |
|---------------------|-------------|-------------------------------|----------------|
| `Dockerfile-linux`  | Linux x64   | `ubuntu:24.04`, native gcc    | Linux          |
| `Dockerfile-windows`| Windows x64 | `ubuntu:24.04` + MinGW-w64   | Linux          |

## Build Configuration

| Setting    | Value    |
|------------|----------|
| RocksDB    | v9.10.0  |
| Linux base | Ubuntu 24.04, native gcc |
| Windows    | Ubuntu 24.04, MinGW-w64 cross-compile |

The Windows build disables optional compression libraries (Snappy, LZ4, Zlib, Zstd, BZ2) to simplify cross-compilation. Enable them if needed by adjusting the CMake flags in `Dockerfile-windows`.