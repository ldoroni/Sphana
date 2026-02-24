# ─────────────────────────────────────────────────────────────
# RocksDB Native Library Builder
# Builds RocksDB shared libraries for Linux and Windows using Docker,
# then copies them to the target runtimes directory.
#
# Dockerfiles used:
#   Dockerfile-linux   - compiles RocksDB from source (Ubuntu)
#   Dockerfile-windows - cross-compiles RocksDB for Windows (Ubuntu + MinGW-w64)
# ─────────────────────────────────────────────────────────────

param(
    [ValidateSet("all", "linux", "windows")]
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

Write-Host "=== RocksDB Native Libraries - Build ===" -ForegroundColor Cyan
Write-Host "  Target runtimes dir: $RuntimesDir" -ForegroundColor Gray
Write-Host ""

# -- Helper: Build Linux ----------------------------------------
function Build-Linux {
    Write-Host "[Linux x64] Building libraries [Dockerfile-linux]..." -ForegroundColor Cyan

    if (Test-Path "$ScriptDir/output-linux") { Remove-Item -Recurse -Force "$ScriptDir/output-linux" }
    docker build -f "$ScriptDir/Dockerfile-linux" --output "type=local,dest=$ScriptDir/output-linux" "$ScriptDir"
    if ($LASTEXITCODE -ne 0) { Write-Error "Linux build failed!"; exit 1 }

    Write-Host "  Linux build complete." -ForegroundColor Green
    Get-ChildItem "$ScriptDir/output-linux/output" | Format-Table Name, Length
}

# -- Helper: Build Windows --------------------------------------
function Build-Windows {
    Write-Host "[Windows x64] Cross-compiling libraries [Dockerfile-windows]..." -ForegroundColor Cyan

    if (Test-Path "$ScriptDir/output-windows") { Remove-Item -Recurse -Force "$ScriptDir/output-windows" }
    docker build -f "$ScriptDir/Dockerfile-windows" --output "type=local,dest=$ScriptDir/output-windows" "$ScriptDir"
    if ($LASTEXITCODE -ne 0) { Write-Error "Windows build failed!"; exit 1 }

    Write-Host "  Windows build complete." -ForegroundColor Green
    Get-ChildItem "$ScriptDir/output-windows/output" | Format-Table Name, Length
}

# -- Dispatch builds based on target ----------------------------
switch ($Target) {
    "linux"   { Build-Linux }
    "windows" { Build-Windows }
    "all"     { Build-Linux; Build-Windows }
}

# -- Copy to runtimes ------------------------------------------

Write-Host "Copying libraries to runtimes directory..." -ForegroundColor Yellow

# Linux
if (Test-Path "$ScriptDir/output-linux/output") {
    New-Item -ItemType Directory -Force -Path "$RuntimesDir/linux-x64/native" | Out-Null
    Copy-Item -Force "$ScriptDir/output-linux/output/librocksdb.so" -Destination "$RuntimesDir/linux-x64/native/"
}

# Windows
if (Test-Path "$ScriptDir/output-windows/output") {
    New-Item -ItemType Directory -Force -Path "$RuntimesDir/win-x64/native" | Out-Null
    Copy-Item -Force "$ScriptDir/output-windows/output/rocksdb.dll" -Destination "$RuntimesDir/win-x64/native/"
}

# -- Cleanup build output dirs ---------------------------------

Write-Host "Cleaning up build output..." -ForegroundColor Yellow

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ScriptDir/output-linux"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ScriptDir/output-windows"

# -- Summary ----------------------------------------------------

Write-Host ""
Write-Host "=== Done! Libraries copied to $RuntimesDir ===" -ForegroundColor Green
Write-Host ""
Write-Host "  Runtimes layout:" -ForegroundColor Gray
Write-Host "    linux-x64/native/  - librocksdb.so" -ForegroundColor Gray
Write-Host "    win-x64/native/    - rocksdb.dll" -ForegroundColor Gray
Write-Host ""
Write-Host "  Usage:" -ForegroundColor Gray
Write-Host "    .\run.ps1                                    # Build all, output to ./target" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -Target linux                      # Build Linux only" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -Target windows                    # Build Windows only" -ForegroundColor DarkGray
Write-Host "    .\run.ps1 -RuntimesDir path\to\runtimes      # Custom output directory" -ForegroundColor DarkGray