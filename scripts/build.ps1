# CallWhisper Build Script (Windows PowerShell)
# Builds the application into a portable Windows executable
#
# Output:
#   - dist/CallWhisper/      : Portable folder (can be run directly)
#   - dist/CallWhisper.zip   : Portable ZIP for distribution
#
# For enterprise deployment, the resulting bundle:
#   - Has ZERO external network dependencies
#   - Uses local whisper.cpp for transcription
#   - Stores all data in a 'data' subdirectory next to the executable
#   - Can be deployed via XCOPY, USB, or network share

$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DistDir = Join-Path $ProjectRoot "dist"
$BuildDir = Join-Path $ProjectRoot "build"
$ReleaseDir = Join-Path $DistDir "CallWhisper"
$ReleaseName = "CallWhisper_Windows_x64"

# Get version from __init__.py
$InitFile = Join-Path $ProjectRoot "src\callwhisper\__init__.py"
if (Test-Path $InitFile) {
    $versionLine = Get-Content $InitFile | Select-String '__version__\s*=\s*"([^"]+)"'
    if ($versionLine) {
        $VERSION = $versionLine.Matches.Groups[1].Value
    } else {
        $VERSION = "1.0.0"
    }
} else {
    $VERSION = "1.0.0"
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CallWhisper Build Script" -ForegroundColor Cyan
Write-Host "  Version: $VERSION" -ForegroundColor Cyan
Write-Host "  Mode: OFFLINE (No Cloud Dependencies)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to project root
Set-Location $ProjectRoot

# Step 0: Pre-flight checks - Verify required vendor binaries
Write-Host "[0/7] Running pre-flight checks..." -ForegroundColor Yellow

$RequiredFiles = @{
    "vendor/ffmpeg.exe" = "Download from ffmpeg.org or run: .\scripts\download-vendor.ps1"
    "vendor/ffprobe.exe" = "Download from ffmpeg.org or run: .\scripts\download-vendor.ps1"
    "vendor/whisper-cli.exe" = "Build from whisper.cpp or run: .\scripts\download-vendor.ps1"
    "models/ggml-medium.en.bin" = "Download from huggingface.co/ggerganov/whisper.cpp or run: .\scripts\download-vendor.ps1"
}

$allFilesPresent = $true
$totalVendorSize = 0

foreach ($file in $RequiredFiles.Keys) {
    $filePath = Join-Path $ProjectRoot $file
    if (Test-Path $filePath) {
        $size = (Get-Item $filePath).Length / 1MB
        $totalVendorSize += $size
        Write-Host "  [OK] $file ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $file" -ForegroundColor Red
        Write-Host "       -> $($RequiredFiles[$file])" -ForegroundColor Yellow
        $allFilesPresent = $false
    }
}

if (-not $allFilesPresent) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  BUILD FAILED: Missing Required Files" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Run the vendor download script first:" -ForegroundColor Yellow
    Write-Host "  .\scripts\download-vendor.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Or manually download the files from:" -ForegroundColor Yellow
    Write-Host "  FFmpeg:      https://github.com/BtbN/FFmpeg-Builds/releases" -ForegroundColor White
    Write-Host "  Whisper.cpp: https://github.com/ggerganov/whisper.cpp/releases" -ForegroundColor White
    Write-Host "  Model:       https://huggingface.co/ggerganov/whisper.cpp" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "  Total vendor size: $([math]::Round($totalVendorSize, 0)) MB" -ForegroundColor Cyan
Write-Host "  All required files present" -ForegroundColor Green
Write-Host ""

# Step 1: Clean previous builds
Write-Host "[1/7] Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path $DistDir) {
    Remove-Item -Recurse -Force $DistDir
}
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}
Write-Host "  Done" -ForegroundColor Green

# Step 2: Check dependencies
Write-Host "[2/7] Checking dependencies..." -ForegroundColor Yellow

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "  ERROR: Python not found!" -ForegroundColor Red
    exit 1
}

# Check PyInstaller
$pyinstaller = python -c "import PyInstaller" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing PyInstaller..." -ForegroundColor Yellow
    pip install pyinstaller
}

Write-Host "  Done" -ForegroundColor Green

# Step 3: Install requirements
Write-Host "[3/7] Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt -q
Write-Host "  Done" -ForegroundColor Green

# Step 4: Run PyInstaller
Write-Host "[4/7] Building with PyInstaller..." -ForegroundColor Yellow

$pyinstallerArgs = @(
    "--name=CallWhisper",
    "--onedir",
    "--windowed",
    "--add-data=static;static",
    "--add-data=config;config",
    "--add-data=src/callwhisper;callwhisper",
    "--hidden-import=uvicorn.logging",
    "--hidden-import=uvicorn.loops",
    "--hidden-import=uvicorn.loops.auto",
    "--hidden-import=uvicorn.protocols",
    "--hidden-import=uvicorn.protocols.http",
    "--hidden-import=uvicorn.protocols.http.auto",
    "--hidden-import=uvicorn.protocols.websockets",
    "--hidden-import=uvicorn.protocols.websockets.auto",
    "--hidden-import=uvicorn.lifespan",
    "--hidden-import=uvicorn.lifespan.on",
    "--hidden-import=fastapi",
    "--hidden-import=starlette",
    "--hidden-import=pydantic",
    "--hidden-import=pydantic_settings",
    "--collect-submodules=uvicorn",
    "--collect-submodules=fastapi",
    "--collect-submodules=starlette",
    "pyinstaller_entrypoint.py"
)

python -m PyInstaller @pyinstallerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: PyInstaller failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Done" -ForegroundColor Green

# Step 5: Copy vendor files and create directories
Write-Host "[5/7] Setting up release directory..." -ForegroundColor Yellow

# Create required directories (data subdirectory for user data)
$dataDir = Join-Path $ReleaseDir "data"
$vendorDir = Join-Path $ReleaseDir "vendor"
$modelsDir = Join-Path $ReleaseDir "models"
$outputDir = Join-Path $dataDir "output"
$configDir = Join-Path $dataDir "config"
$logsDir = Join-Path $dataDir "logs"
$checkpointsDir = Join-Path $dataDir "checkpoints"

# Create all directories
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
New-Item -ItemType Directory -Force -Path $configDir | Out-Null
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointsDir | Out-Null

Write-Host "  Created directory structure" -ForegroundColor Gray

# Copy vendor binaries (pre-flight checks ensure these exist)
$sourceVendor = Join-Path $ProjectRoot "vendor"
Copy-Item -Path "$sourceVendor\*" -Destination $vendorDir -Recurse -Force
Write-Host "  Copied vendor binaries (ffmpeg, ffprobe, whisper-cli)" -ForegroundColor Gray

# Copy models (pre-flight checks ensure these exist)
$sourceModels = Join-Path $ProjectRoot "models"
Copy-Item -Path "$sourceModels\*" -Destination $modelsDir -Recurse -Force
Write-Host "  Copied whisper model (ggml-medium.en.bin)" -ForegroundColor Gray

# Ensure config exists
$sourceConfig = Join-Path $ProjectRoot "config\config.json"
$destConfig = Join-Path $configDir "config.json"
if (Test-Path $sourceConfig) {
    Copy-Item $sourceConfig $destConfig -Force
    Write-Host "  Copied configuration" -ForegroundColor Gray
}

Write-Host "  Done" -ForegroundColor Green

# Step 6: Create version.json for portable mode detection
Write-Host "[6/7] Creating version.json..." -ForegroundColor Yellow

$versionInfo = @{
    version = $VERSION
    build_date = (Get-Date -Format "yyyy-MM-dd")
    build_time = (Get-Date -Format "HH:mm:ss")
    offline_mode = $true
    network_guard = "enabled"
    external_api_calls = "none"
    transcription_engine = "whisper.cpp (local)"
    deployment_type = "portable"
}

$versionJsonPath = Join-Path $ReleaseDir "version.json"
$versionInfo | ConvertTo-Json -Depth 3 | Out-File -FilePath $versionJsonPath -Encoding UTF8

Write-Host "  Created version.json" -ForegroundColor Gray
Write-Host "  Done" -ForegroundColor Green

# Step 7: Create release ZIP
Write-Host "[7/7] Creating release ZIP..." -ForegroundColor Yellow

$zipPath = Join-Path $DistDir "$ReleaseName.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path $ReleaseDir -DestinationPath $zipPath

Write-Host "  Done" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "  Version: $VERSION" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Release folder: $ReleaseDir" -ForegroundColor White
Write-Host "Release ZIP:    $zipPath" -ForegroundColor White
Write-Host ""
Write-Host "Directory Structure:" -ForegroundColor Cyan
Write-Host "  CallWhisper/" -ForegroundColor White
Write-Host "    CallWhisper.exe   - Main application" -ForegroundColor Gray
Write-Host "    version.json      - Version info (enables portable mode)" -ForegroundColor Gray
Write-Host "    vendor/           - FFmpeg and Whisper binaries" -ForegroundColor Gray
Write-Host "    models/           - Whisper transcription models" -ForegroundColor Gray
Write-Host "    data/             - User data directory" -ForegroundColor Gray
Write-Host "      config/         - Configuration files" -ForegroundColor Gray
Write-Host "      output/         - Recording output" -ForegroundColor Gray
Write-Host "      logs/           - Application logs" -ForegroundColor Gray
Write-Host ""
Write-Host "Enterprise Deployment:" -ForegroundColor Cyan
Write-Host "  - FULLY OFFLINE: No external network dependencies" -ForegroundColor Green
Write-Host "  - ZERO cloud API calls (no Claude, OpenAI, etc.)" -ForegroundColor Green
Write-Host "  - Local whisper.cpp transcription engine" -ForegroundColor Green
Write-Host "  - Can be deployed via XCOPY, USB, or network share" -ForegroundColor Green
Write-Host ""
Write-Host "Bundled vendor files (verified):" -ForegroundColor Cyan
Write-Host "  - ffmpeg.exe, ffprobe.exe (audio processing)" -ForegroundColor Gray
Write-Host "  - whisper-cli.exe (transcription engine)" -ForegroundColor Gray
Write-Host "  - ggml-medium.en.bin (Whisper model, ~1.5 GB)" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Test: .\dist\CallWhisper\CallWhisper.exe" -ForegroundColor Gray
Write-Host "  2. Distribute ZIP to end users or use MSI installer" -ForegroundColor Gray
Write-Host ""
Write-Host "For MSI installer, run: .\scripts\build-msi.ps1" -ForegroundColor Yellow
Write-Host ""
