# CallWhisper Vendor Binary Download Script (Windows PowerShell)
#
# Downloads required third-party binaries for building CallWhisper MSI
# Run this script BEFORE running build.ps1
#
# Downloads:
#   - FFmpeg (GPL licensed) - Audio processing
#   - Whisper.cpp CLI (MIT licensed) - Local transcription engine
#   - ggml-medium.en.bin (MIT licensed) - Whisper model weights
#
# Total download size: ~1.6 GB
# Required disk space: ~2 GB (downloads + extracted)

$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VendorDir = Join-Path $ProjectRoot "vendor"
$ModelsDir = Join-Path $ProjectRoot "models"
$TempDir = Join-Path $env:TEMP "callwhisper_vendor"

# Download URLs
# FFmpeg: Static build from BtbN (GPL licensed, includes all codecs)
$FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$FfmpegZip = Join-Path $TempDir "ffmpeg.zip"

# Whisper.cpp: Pre-built Windows binaries
# Using v1.7.4 (latest stable with CUDA support)
$WhisperVersion = "v1.7.4"
$WhisperUrl = "https://github.com/ggerganov/whisper.cpp/releases/download/$WhisperVersion/whisper-bin-x64.zip"
$WhisperZip = Join-Path $TempDir "whisper.zip"

# Whisper model: ggml-medium.en (English-only, ~1.5 GB)
# Faster and more accurate for English than multilingual
$ModelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin"
$ModelPath = Join-Path $ModelsDir "ggml-medium.en.bin"

# Expected file sizes (approximate, for verification)
$ExpectedSizes = @{
    "ffmpeg.exe" = 100MB      # ~100-130 MB
    "whisper-cli.exe" = 5MB   # ~5-10 MB (without CUDA)
    "ggml-medium.en.bin" = 1400MB  # ~1.5 GB
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CallWhisper Vendor Download Script" -ForegroundColor Cyan
Write-Host "  Downloads: FFmpeg, Whisper.cpp, Model" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create directories
Write-Host "[1/5] Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
Write-Host "  Created: $VendorDir" -ForegroundColor Gray
Write-Host "  Created: $ModelsDir" -ForegroundColor Gray
Write-Host "  Done" -ForegroundColor Green
Write-Host ""

# Function to download with progress
function Download-WithProgress {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Description
    )

    Write-Host "  Downloading $Description..." -ForegroundColor Gray
    Write-Host "  URL: $Url" -ForegroundColor DarkGray

    try {
        # Use BITS for large downloads (shows progress)
        if ((Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue)) {
            Start-BitsTransfer -Source $Url -Destination $OutFile -Description $Description
        } else {
            # Fallback to Invoke-WebRequest
            $ProgressPreference = 'SilentlyContinue'  # Speed up download
            Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
            $ProgressPreference = 'Continue'
        }

        if (Test-Path $OutFile) {
            $size = (Get-Item $OutFile).Length / 1MB
            Write-Host "  Downloaded: $([math]::Round($size, 1)) MB" -ForegroundColor Gray
            return $true
        }
    } catch {
        Write-Host "  ERROR: Download failed - $_" -ForegroundColor Red
        return $false
    }
    return $false
}

# Download FFmpeg
Write-Host "[2/5] Downloading FFmpeg..." -ForegroundColor Yellow

$ffmpegExe = Join-Path $VendorDir "ffmpeg.exe"
$ffprobeExe = Join-Path $VendorDir "ffprobe.exe"

if ((Test-Path $ffmpegExe) -and (Test-Path $ffprobeExe)) {
    Write-Host "  FFmpeg already exists, skipping download" -ForegroundColor Gray
} else {
    if (Download-WithProgress -Url $FfmpegUrl -OutFile $FfmpegZip -Description "FFmpeg") {
        Write-Host "  Extracting FFmpeg..." -ForegroundColor Gray

        # Extract to temp, then copy needed files
        $extractDir = Join-Path $TempDir "ffmpeg_extract"
        Expand-Archive -Path $FfmpegZip -DestinationPath $extractDir -Force

        # Find the bin directory (structure: ffmpeg-xxx-win64-gpl/bin/)
        $binDir = Get-ChildItem -Path $extractDir -Recurse -Directory -Filter "bin" | Select-Object -First 1

        if ($binDir) {
            Copy-Item -Path (Join-Path $binDir.FullName "ffmpeg.exe") -Destination $VendorDir -Force
            Copy-Item -Path (Join-Path $binDir.FullName "ffprobe.exe") -Destination $VendorDir -Force
            Write-Host "  Extracted ffmpeg.exe and ffprobe.exe" -ForegroundColor Gray
        } else {
            Write-Host "  ERROR: Could not find FFmpeg binaries in archive" -ForegroundColor Red
            exit 1
        }

        # Cleanup
        Remove-Item -Recurse -Force $extractDir
        Remove-Item -Force $FfmpegZip
    } else {
        Write-Host "  ERROR: Failed to download FFmpeg" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  Done" -ForegroundColor Green
Write-Host ""

# Download Whisper.cpp
Write-Host "[3/5] Downloading Whisper.cpp CLI..." -ForegroundColor Yellow

$whisperCli = Join-Path $VendorDir "whisper-cli.exe"
# Alternative name used in some releases
$mainExe = Join-Path $VendorDir "main.exe"

if (Test-Path $whisperCli) {
    Write-Host "  Whisper CLI already exists, skipping download" -ForegroundColor Gray
} else {
    if (Download-WithProgress -Url $WhisperUrl -OutFile $WhisperZip -Description "Whisper.cpp $WhisperVersion") {
        Write-Host "  Extracting Whisper.cpp..." -ForegroundColor Gray

        $extractDir = Join-Path $TempDir "whisper_extract"
        Expand-Archive -Path $WhisperZip -DestinationPath $extractDir -Force

        # Look for main.exe or whisper-cli.exe
        $whisperBin = Get-ChildItem -Path $extractDir -Recurse -Filter "main.exe" | Select-Object -First 1
        if (-not $whisperBin) {
            $whisperBin = Get-ChildItem -Path $extractDir -Recurse -Filter "whisper-cli.exe" | Select-Object -First 1
        }
        if (-not $whisperBin) {
            $whisperBin = Get-ChildItem -Path $extractDir -Recurse -Filter "whisper.exe" | Select-Object -First 1
        }

        if ($whisperBin) {
            # Always copy as whisper-cli.exe for consistent naming
            Copy-Item -Path $whisperBin.FullName -Destination $whisperCli -Force
            Write-Host "  Extracted as whisper-cli.exe" -ForegroundColor Gray
        } else {
            Write-Host "  ERROR: Could not find Whisper binary in archive" -ForegroundColor Red
            Write-Host "  Archive contents:" -ForegroundColor Yellow
            Get-ChildItem -Path $extractDir -Recurse | ForEach-Object { Write-Host "    $_" }
            exit 1
        }

        # Cleanup
        Remove-Item -Recurse -Force $extractDir
        Remove-Item -Force $WhisperZip
    } else {
        Write-Host "  ERROR: Failed to download Whisper.cpp" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  Done" -ForegroundColor Green
Write-Host ""

# Download Whisper Model
Write-Host "[4/5] Downloading Whisper model (ggml-medium.en)..." -ForegroundColor Yellow
Write-Host "  This is a large file (~1.5 GB), please be patient..." -ForegroundColor DarkGray

if (Test-Path $ModelPath) {
    $existingSize = (Get-Item $ModelPath).Length / 1MB
    if ($existingSize -gt 1400) {
        Write-Host "  Model already exists ($([math]::Round($existingSize, 0)) MB), skipping download" -ForegroundColor Gray
    } else {
        Write-Host "  Existing model file appears incomplete, re-downloading..." -ForegroundColor Yellow
        Remove-Item -Force $ModelPath
    }
}

if (-not (Test-Path $ModelPath)) {
    if (Download-WithProgress -Url $ModelUrl -OutFile $ModelPath -Description "ggml-medium.en.bin") {
        $modelSize = (Get-Item $ModelPath).Length / 1MB
        Write-Host "  Model downloaded: $([math]::Round($modelSize, 0)) MB" -ForegroundColor Gray
    } else {
        Write-Host "  ERROR: Failed to download Whisper model" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Manual download instructions:" -ForegroundColor Yellow
        Write-Host "  1. Visit: https://huggingface.co/ggerganov/whisper.cpp" -ForegroundColor White
        Write-Host "  2. Download: ggml-medium.en.bin" -ForegroundColor White
        Write-Host "  3. Place in: $ModelsDir" -ForegroundColor White
        exit 1
    }
}
Write-Host "  Done" -ForegroundColor Green
Write-Host ""

# Verify all files
Write-Host "[5/5] Verifying downloads..." -ForegroundColor Yellow

$allGood = $true

# Check FFmpeg
if (Test-Path $ffmpegExe) {
    $size = (Get-Item $ffmpegExe).Length / 1MB
    Write-Host "  [OK] ffmpeg.exe ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] ffmpeg.exe" -ForegroundColor Red
    $allGood = $false
}

# Check FFprobe
if (Test-Path $ffprobeExe) {
    $size = (Get-Item $ffprobeExe).Length / 1MB
    Write-Host "  [OK] ffprobe.exe ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] ffprobe.exe" -ForegroundColor Red
    $allGood = $false
}

# Check Whisper CLI
if (Test-Path $whisperCli) {
    $size = (Get-Item $whisperCli).Length / 1MB
    Write-Host "  [OK] whisper-cli.exe ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] whisper-cli.exe" -ForegroundColor Red
    $allGood = $false
}

# Check Model
if (Test-Path $ModelPath) {
    $size = (Get-Item $ModelPath).Length / 1MB
    if ($size -gt 1400) {
        Write-Host "  [OK] ggml-medium.en.bin ($([math]::Round($size, 0)) MB)" -ForegroundColor Green
    } else {
        Write-Host "  [INCOMPLETE] ggml-medium.en.bin ($([math]::Round($size, 0)) MB - expected ~1500 MB)" -ForegroundColor Yellow
        $allGood = $false
    }
} else {
    Write-Host "  [MISSING] ggml-medium.en.bin" -ForegroundColor Red
    $allGood = $false
}

Write-Host ""

# Cleanup temp directory
if (Test-Path $TempDir) {
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}

# Summary
if ($allGood) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  All Vendor Files Ready!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Vendor directory: $VendorDir" -ForegroundColor White
    Write-Host "Models directory: $ModelsDir" -ForegroundColor White
    Write-Host ""
    Write-Host "Next step: Run build.ps1 to create the application" -ForegroundColor Yellow
    Write-Host "  .\scripts\build.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Some Files Missing or Incomplete" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the errors above and try again." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Manual download sources:" -ForegroundColor Cyan
    Write-Host "  FFmpeg:      https://github.com/BtbN/FFmpeg-Builds/releases" -ForegroundColor White
    Write-Host "  Whisper.cpp: https://github.com/ggerganov/whisper.cpp/releases" -ForegroundColor White
    Write-Host "  Model:       https://huggingface.co/ggerganov/whisper.cpp" -ForegroundColor White
    Write-Host ""
    exit 1
}
