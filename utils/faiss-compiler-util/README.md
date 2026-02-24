# FAISS Native Library Compiler

Builds FAISS shared libraries (`libfaiss_c.so` / `faiss_c.dll` and `libfaiss.so` / `faiss.dll`) from source using Docker, including optional **CUDA GPU support**.

These native libraries are used by the `Sphana.DataStore` .NET service via P/Invoke, supporting .NET AOT compilation.

## Prerequisites

- Docker Desktop
- For Linux CUDA build: Docker must be able to pull `nvidia/cuda` base images (no GPU required for **building**)
- For Windows CUDA build: Docker must be in **Windows container mode**

## Quick Start

```powershell
.\utils\faiss-compiler-util\run.ps1 -RuntimesDir "..\..\services\Sphana\Services\Sphana.DataStore\runtimes" -Target cpu
```

### More Options

```powershell
# Build all variants (output to ./target)
.\utils\faiss-compiler-util\run.ps1

# Build CPU-only (Linux + Windows)
.\utils\faiss-compiler-util\run.ps1 -Target cpu

# Build CUDA/GPU (Linux only)
.\utils\faiss-compiler-util\run.ps1 -Target cuda

# Build individual targets
.\utils\faiss-compiler-util\run.ps1 -Target linux          # Linux CPU only
.\utils\faiss-compiler-util\run.ps1 -Target linux-cuda     # Linux CUDA only
.\utils\faiss-compiler-util\run.ps1 -Target windows        # Windows CPU only
.\utils\faiss-compiler-util\run.ps1 -Target windows-cuda   # Windows CUDA only (requires Windows containers)

# Custom output directory
.\utils\faiss-compiler-util\run.ps1 -RuntimesDir "..\..\services\Sphana\Services\Sphana.DataStore\runtimes"
```

## Parameters

| Parameter      | Default     | Description                                                              |
|----------------|-------------|--------------------------------------------------------------------------|
| `-Target`      | `all`       | Build target: `all`, `cpu`, `cuda`, `linux`, `linux-cuda`, `windows`, `windows-cuda` |
| `-RuntimesDir` | `./target`  | Output directory for the runtimes folder layout                          |

## Dockerfiles

| Dockerfile                | Platform       | GPU  | Base Image                              | Container Mode |
|---------------------------|----------------|------|-----------------------------------------|----------------|
| `Dockerfile-linux`        | Linux x64      | No   | `ubuntu:24.04`                          | Linux           |
| `Dockerfile-linux-cuda`   | Linux x64      | Yes  | `nvidia/cuda:12.8.0-devel-ubuntu24.04`  | Linux           |
| `Dockerfile-windows`      | Windows x64    | No   | `ubuntu:24.04` + MinGW-w64             | Linux           |
| `Dockerfile-windows-cuda` | Windows x64    | Yes  | `windows/servercore:ltsc2022` + MSVC    | **Windows**     |

## Output

Libraries are copied to the runtimes directory in the standard .NET native library layout:

```
<RuntimesDir>/
├── linux-x64/native/
│   ├── libfaiss_c.so        # CPU-only
│   └── libfaiss.so
├── linux-x64-cuda/native/
│   ├── libfaiss_c.so        # GPU-enabled (CUDA 12.8)
│   └── libfaiss.so
├── win-x64/native/
│   ├── faiss_c.dll           # CPU-only
│   └── faiss.dll
└── win-x64-cuda/native/
    ├── faiss_c.dll           # GPU-enabled (CUDA 12.8)
    ├── faiss.dll
    └── libopenblas.dll       # OpenBLAS runtime dependency
```

To deploy to the `Sphana.DataStore` service:

```powershell
.\utils\faiss-compiler-util\run.ps1 -RuntimesDir "..\..\services\Sphana\Services\Sphana.DataStore\runtimes"
```

## CUDA GPU Support

### Supported GPU Architectures

Both Linux and Windows CUDA builds target these compute capabilities:

| Arch | Generation         | Example GPUs                    |
|------|--------------------|---------------------------------|
| 70   | Volta              | V100                            |
| 75   | Turing             | T4, RTX 2080                    |
| 80   | Ampere             | A100, RTX 3090                  |
| 86   | Ampere             | RTX 3060/3070/3080              |
| 89   | Ada Lovelace       | RTX 4090, L40                   |
| 90   | Hopper             | H100                            |

### Linux CUDA Build

Uses `Dockerfile-linux-cuda` with `nvidia/cuda:12.8.0-devel-ubuntu24.04` base image. Runs in standard Linux container mode — no GPU required for building.

### Windows CUDA Build

Uses `Dockerfile-windows-cuda` with `mcr.microsoft.com/windows/servercore:ltsc2022` base image. This Dockerfile:

1. Installs Visual Studio 2022 Build Tools (C++ workload)
2. Installs CUDA Toolkit 12.8
3. Downloads OpenBLAS
4. Clones and builds FAISS with `-DFAISS_ENABLE_GPU=ON`

**Requirements:**
- Docker must be in **Windows container mode** (not the default Linux container mode)
- Switch via: Docker Desktop → right-click tray icon → "Switch to Windows containers..."
- Or from CLI: `& 'C:\Program Files\Docker\Docker\DockerCli.exe' -SwitchDaemon`
- No GPU required for building — only the CUDA toolkit (downloaded during build)

**Note:** After building the Windows CUDA image, switch Docker back to Linux container mode for the other builds.

### Important Notes

- **Building** the CUDA variant does **not** require a GPU — only the CUDA dev toolkit
- **Running** the GPU-enabled libraries requires an NVIDIA GPU with compatible drivers (CUDA 12.x)
- The GPU build is significantly larger than CPU-only due to embedded CUDA kernels for multiple architectures
- On Windows, CUDA runtime DLLs (`cudart64_*.dll`, `cublas64_*.dll`) must be available at runtime

## Build Configuration

| Setting       | Value    |
|---------------|----------|
| FAISS version | v1.9.0   |
| BLAS          | OpenBLAS |
| CUDA          | 12.8     |