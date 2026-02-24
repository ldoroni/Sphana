# -----------------------------------------------------------------
# FAISS C API - Build & Extract Script (PowerShell)
# Uses separate Dockerfiles for Linux, Linux+CUDA, and Windows builds,
# then copies all native libraries to the target runtimes directory.
#
# Dockerfiles used:
#   Dockerfile-linux        - compiles FAISS from source (Ubuntu, CPU-only)
#   Dockerfile-linux-cuda   - compiles FAISS with CUDA 12.8 GPU support
#   Dockerfile-windows      - cross-compiles FAISS from source (Ubuntu + MinGW-w64)
#   Dockerfile-windows-cuda - builds FAISS with CUDA (Windows container + MSVC)
#                             ** Requires Docker in Windows container mode **
# -----------------------------------------------------------------

param(
    [ValidateSet("all", "cpu", "cuda", "linux", "linux-cuda", "windows", "windows-cuda")]
    [string]$Target = "all",

    [string]$RuntimesDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Default RuntimesDir to ./target relative to the script
if (-not $RuntimesDir) {
    $RuntimesDir = Join-Path $ScriptDir "target"
}

# Resolve relative paths relative to the script directory (not CWD)
if (-not [System.IO.Path]::IsPathRooted($RuntimesDir)) {
    $RuntimesDir = Join-Path $ScriptDir $RuntimesDir
}
$RuntimesDir = [System.IO.Path]::GetFullPath($RuntimesDir)

Write-Host "=== FAISS Native Libraries - Build ===" -ForegroundColor Cyan
Write-Host "  Target runtimes dir: $RuntimesDir" -ForegroundColor Gray
Write-Host ""

# -- Helper: Build Linux CPU ------------------------------------
function Build-LinuxCpu {
    Write-Host "[Linux x64 CPU] Building libraries [Dockerfile-linux]..." -ForegroundColor Cyan

    if (Test-Path "$ScriptDir/output-linux") { Remove-Item -Recurse -Force "$ScriptDir/output-linux" }
    docker build -f "$ScriptDir/Dockerfile-linux" --output "type=local,dest=$ScriptDir/output-linux" "$ScriptDir"
    if ($LASTEXITCODE -ne 0) { Write-Error "Linux CPU build failed!"; exit 1 }

    Write-Host "  Linux CPU build complete." -ForegroundColor Green
    Write-Host ""
}

# -- Helper: Build Linux CUDA -----------------------------------
function Build-LinuxCuda {
    Write-Host "[Linux x64 CUDA] Building GPU-enabled libraries [Dockerfile-linux-cuda]..." -ForegroundColor Cyan
    Write-Host "  NOTE: This builds for CUDA 12.8. No GPU needed for compilation." -ForegroundColor DarkGray

    if (Test-Path "$ScriptDir/output-linux-cuda") { Remove-Item -Recurse -Force "$ScriptDir/output-linux-cuda" }
    docker build -f "$ScriptDir/Dockerfile-linux-cuda" --output "type=local,dest=$ScriptDir/output-linux-cuda" "$ScriptDir"
    if ($LASTEXITCODE -ne 0) { Write-Error "Linux CUDA build failed!"; exit 1 }

    Write-Host "  Linux CUDA build complete." -ForegroundColor Green
    Write-Host ""
}

# -- Helper: Build Windows CPU ----------------------------------
function Build-Windows {
    Write-Host "[Windows x64 CPU] Cross-compiling libraries [Dockerfile-windows]..." -ForegroundColor Cyan

    if (Test-Path "$ScriptDir/output-windows") { Remove-Item -Recurse -Force "$ScriptDir/output-windows" }
    docker build -f "$ScriptDir/Dockerfile-windows" --output "type=local,dest=$ScriptDir/output-windows" "$ScriptDir"
    if ($LASTEXITCODE -ne 0) { Write-Error "Windows build failed!"; exit 1 }

    Write-Host "  Windows build complete." -ForegroundColor Green
    Write-Host ""
}

# -- Helper: Build Windows CUDA ---------------------------------
function Build-WindowsCuda {
    Write-Host "[Windows x64 CUDA] Building GPU-enabled libraries [Dockerfile-windows-cuda]..." -ForegroundColor Cyan
    Write-Host "  IMPORTANT: Docker must be in Windows container mode!" -ForegroundColor Yellow
    Write-Host "  Switch via: Docker Desktop > Settings > Docker Engine > Windows containers" -ForegroundColor DarkGray
    Write-Host ""

    $containerName = "faiss-win-cuda-extract"
    $imageName = "faiss-win-cuda"

    # Build the image
    docker build -f "$ScriptDir/Dockerfile-windows-cuda" -t $imageName "$ScriptDir"
    if ($LASTEXITCODE -ne 0) { Write-Error "Windows CUDA build failed!"; exit 1 }

    # Extract artifacts via docker cp (Windows containers don't support --output)
    docker rm $containerName -f 2>$null
    docker create --name $containerName $imageName
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create extraction container!"; exit 1 }

    if (Test-Path "$ScriptDir/output-windows-cuda") { Remove-Item -Recurse -Force "$ScriptDir/output-windows-cuda" }
    New-Item -ItemType Directory -Force -Path "$ScriptDir/output-windows-cuda/output" | Out-Null

    docker cp "${containerName}:C:\output\faiss.dll" "$ScriptDir/output-windows-cuda/output/faiss.dll"
    docker cp "${containerName}:C:\output\faiss_c.dll" "$ScriptDir/output-windows-cuda/output/faiss_c.dll"
    docker cp "${containerName}:C:\output\libopenblas.dll" "$ScriptDir/output-windows-cuda/output/libopenblas.dll" 2>$null

    docker rm $containerName -f

    Write-Host "  Windows CUDA build complete." -ForegroundColor Green
    Write-Host ""
}

# -- Dispatch builds based on target ----------------------------
switch ($Target) {
    "linux"        { Build-LinuxCpu }
    "linux-cuda"   { Build-LinuxCuda }
    "cuda"         { Build-LinuxCuda }
    "windows"      { Build-Windows }
    "windows-cuda" { Build-WindowsCuda }
    "cpu"          { Build-LinuxCpu; Build-Windows }
    "all"          { Build-LinuxCpu; Build-LinuxCuda; Build-Windows }
}

# -- Copy to runtimes ------------------------------------------

Write-Host "Copying libraries to runtimes directory..." -ForegroundColor Yellow

# Linux CPU
if (Test-Path "$ScriptDir/output-linux/output") {
    New-Item -ItemType Directory -Force -Path "$RuntimesDir/linux-x64/native" | Out-Null
    Copy-Item -Force "$ScriptDir/output-linux/output/libfaiss_c.so" -Destination "$RuntimesDir/linux-x64/native/"
    Copy-Item -Force "$ScriptDir/output-linux/output/libfaiss.so"   -Destination "$RuntimesDir/linux-x64/native/"
}

# Linux CUDA (GPU)
if (Test-Path "$ScriptDir/output-linux-cuda/output") {
    New-Item -ItemType Directory -Force -Path "$RuntimesDir/linux-x64-cuda/native" | Out-Null
    Copy-Item -Force "$ScriptDir/output-linux-cuda/output/libfaiss_c.so" -Destination "$RuntimesDir/linux-x64-cuda/native/"
    Copy-Item -Force "$ScriptDir/output-linux-cuda/output/libfaiss.so"   -Destination "$RuntimesDir/linux-x64-cuda/native/"
}

# Windows CPU
if (Test-Path "$ScriptDir/output-windows/output") {
    New-Item -ItemType Directory -Force -Path "$RuntimesDir/win-x64/native" | Out-Null
    Copy-Item -Force "$ScriptDir/output-windows/output/faiss_c.dll" -Destination "$RuntimesDir/win-x64/native/"
    Copy-Item -Force "$ScriptDir/output-windows/output/faiss.dll"   -Destination "$RuntimesDir/win-x64/native/"
}

# Windows CUDA (GPU)
if (Test-Path "$ScriptDir/output-windows-cuda/output") {
    New-Item -ItemType Directory -Force -Path "$RuntimesDir/win-x64-cuda/native" | Out-Null
    Copy-Item -Force "$ScriptDir/output-windows-cuda/output/faiss_c.dll" -Destination "$RuntimesDir/win-x64-cuda/native/"
    Copy-Item -Force "$ScriptDir/output-windows-cuda/output/faiss.dll"   -Destination "$RuntimesDir/win-x64-cuda/native/"
    if (Test-Path "$ScriptDir/output-windows-cuda/output/libopenblas.dll") {
        Copy-Item -Force "$ScriptDir/output-windows-cuda/output/libopenblas.dll" -Destination "$RuntimesDir/win-x64-cuda/native/"
    }
}

# -- Cleanup build output dirs ---------------------------------

Write-Host "Cleaning up build output..." -ForegroundColor Yellow

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ScriptDir/output-linux"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ScriptDir/output-linux-cuda"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ScriptDir/output-windows"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ScriptDir/output-windows-cuda"

# -- Summary ----------------------------------------------------

Write-Host ""
Write-Host "=== Done! Libraries copied to $RuntimesDir ===" -ForegroundColor Green
Write-Host ""
Write-Host "  Runtimes layout:" -ForegroundColor Gray
Write-Host "    linux-x64/native/       - CPU-only (libfaiss_c.so, libfaiss.so)" -ForegroundColor Gray
Write-Host "    linux-x64-cuda/native/  - GPU/CUDA 12.8 (libfaiss_c.so, libfaiss.so)" -ForegroundColor Gray
Write-Host "    win-x64/native/         - CPU-only (faiss_c.dll, faiss.dll)" -ForegroundColor Gray
Write-Host "    win-x64-cuda/native/    - GPU/CUDA 12.8 (faiss_c.dll, faiss.dll)" -ForegroundColor Gray
Write-Host ""
Write-Host "  Usage:" -ForegroundColor Gray
Write-Host "    .\run.ps1                                    # Build all, output to ./target" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -Target cpu                        # Build CPU-only (Linux + Windows)" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -Target cuda                       # Build CUDA/GPU (Linux only)" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -Target linux                      # Build Linux CPU only" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -Target linux-cuda                 # Build Linux CUDA only" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -Target windows                    # Build Windows CPU only" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -Target windows-cuda               # Build Windows CUDA only (requires Windows containers)" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -RuntimesDir path\to\runtimes      # Custom output directory" -ForegroundColor DarkGray