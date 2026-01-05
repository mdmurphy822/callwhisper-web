# CallWhisper Development Run Script (Windows PowerShell)
# Runs the application in development mode

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Starting CallWhisper in development mode..." -ForegroundColor Cyan
Write-Host ""

# Change to project root
Set-Location $ProjectRoot

# Check if virtual environment exists
$venvPath = Join-Path $ProjectRoot ".venv"
$venvActivate = Join-Path $venvPath "Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venvActivate
}

# Add src to PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

# Run the application
Write-Host "Starting server..." -ForegroundColor Green
Write-Host ""

python -m uvicorn callwhisper.main:app --reload --host 127.0.0.1 --port 8765
