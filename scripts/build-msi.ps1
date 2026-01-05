# CallWhisper MSI Build Script (Windows PowerShell)
# Builds an MSI installer for enterprise deployment
#
# Prerequisites:
#   - WiX Toolset 3.11+ installed (https://wixtoolset.org/)
#   - Portable build already created via build.ps1
#
# Output:
#   - dist/CallWhisper-Setup.msi
#
# Enterprise Deployment:
#   - Silent install: msiexec /i CallWhisper-Setup.msi /qn
#   - SCCM/GPO: Deploy MSI via standard methods
#   - Upgrades: Same UpgradeCode allows in-place upgrades

$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DistDir = Join-Path $ProjectRoot "dist"
$InstallerDir = Join-Path $ProjectRoot "installer"
$SourceDir = Join-Path $DistDir "CallWhisper"
$ObjDir = Join-Path $InstallerDir "obj"

# WiX Toolset paths (common installation locations)
$WixPaths = @(
    "C:\Program Files (x86)\WiX Toolset v3.14\bin",
    "C:\Program Files (x86)\WiX Toolset v3.11\bin",
    "C:\Program Files\WiX Toolset v3.14\bin",
    "C:\Program Files\WiX Toolset v3.11\bin",
    "$env:WIX\bin"
)

# Find WiX installation
$WixPath = $null
foreach ($path in $WixPaths) {
    if (Test-Path $path) {
        $WixPath = $path
        break
    }
}

if (-not $WixPath) {
    Write-Host "ERROR: WiX Toolset not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install WiX Toolset from: https://wixtoolset.org/releases/" -ForegroundColor Yellow
    Write-Host "Or set the WIX environment variable to your installation path." -ForegroundColor Yellow
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CallWhisper MSI Build Script" -ForegroundColor Cyan
Write-Host "  WiX Path: $WixPath" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if portable build exists
if (-not (Test-Path $SourceDir)) {
    Write-Host "ERROR: Portable build not found at $SourceDir" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run build.ps1 first to create the portable build." -ForegroundColor Yellow
    exit 1
}

# Check for CallWhisper.exe
$exePath = Join-Path $SourceDir "CallWhisper.exe"
if (-not (Test-Path $exePath)) {
    Write-Host "ERROR: CallWhisper.exe not found in build directory" -ForegroundColor Red
    exit 1
}

Write-Host "[1/5] Preparing build directory..." -ForegroundColor Yellow

# Create/clean object directory
if (Test-Path $ObjDir) {
    Remove-Item -Recurse -Force $ObjDir
}
New-Item -ItemType Directory -Force -Path $ObjDir | Out-Null

Write-Host "  Done" -ForegroundColor Green

# Step 2: Create license file if not exists
Write-Host "[2/5] Creating license file..." -ForegroundColor Yellow

$licenseFile = Join-Path $InstallerDir "license.rtf"
if (-not (Test-Path $licenseFile)) {
    $licenseContent = @"
{\rtf1\ansi\deff0
{\fonttbl{\f0 Arial;}}
\f0\fs20
\b CallWhisper License Agreement\b0\par
\par
This software is provided "as is" without warranty of any kind.\par
\par
\b Offline Operation\b0\par
This application operates entirely offline with no external network dependencies.\par
All audio processing and transcription occurs locally using whisper.cpp.\par
\par
\b Data Storage\b0\par
All recordings and transcriptions are stored locally in the data folder.\par
No data is transmitted to external servers.\par
}
"@
    $licenseContent | Out-File -FilePath $licenseFile -Encoding ASCII
}

Write-Host "  Done" -ForegroundColor Green

# Step 3: Generate component files using heat.exe
Write-Host "[3/5] Harvesting components with heat.exe..." -ForegroundColor Yellow

$heatExe = Join-Path $WixPath "heat.exe"

# Harvest main application files (excluding vendor and models)
$productWxs = Join-Path $ObjDir "Product.wxs"
& $heatExe dir $SourceDir `
    -cg ProductComponents `
    -dr INSTALLFOLDER `
    -var "var.SourceDir" `
    -ag -sfrag -srd -sreg -scom `
    -out $productWxs `
    -t (Join-Path $InstallerDir "exclude-subdirs.xslt") 2>$null

# If XSLT doesn't exist, harvest everything and we'll filter manually
if (-not (Test-Path $productWxs) -or (Get-Item $productWxs).Length -eq 0) {
    & $heatExe dir $SourceDir `
        -cg ProductComponents `
        -dr INSTALLFOLDER `
        -var "var.SourceDir" `
        -ag -sfrag -srd -sreg -scom `
        -out $productWxs
}

# Harvest vendor directory
$vendorDir = Join-Path $SourceDir "vendor"
$vendorWxs = Join-Path $ObjDir "Vendor.wxs"
if (Test-Path $vendorDir) {
    & $heatExe dir $vendorDir `
        -cg VendorComponents `
        -dr VendorFolder `
        -var "var.VendorDir" `
        -ag -sfrag -srd -sreg -scom `
        -out $vendorWxs
} else {
    # Create empty component group
    $emptyVendor = @"
<?xml version="1.0" encoding="utf-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
    <Fragment>
        <ComponentGroup Id="VendorComponents" />
    </Fragment>
</Wix>
"@
    $emptyVendor | Out-File -FilePath $vendorWxs -Encoding UTF8
}

# Harvest models directory
$modelsDir = Join-Path $SourceDir "models"
$modelsWxs = Join-Path $ObjDir "Models.wxs"
if (Test-Path $modelsDir) {
    & $heatExe dir $modelsDir `
        -cg ModelComponents `
        -dr ModelsFolder `
        -var "var.ModelsDir" `
        -ag -sfrag -srd -sreg -scom `
        -out $modelsWxs
} else {
    # Create empty component group
    $emptyModels = @"
<?xml version="1.0" encoding="utf-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
    <Fragment>
        <ComponentGroup Id="ModelComponents" />
    </Fragment>
</Wix>
"@
    $emptyModels | Out-File -FilePath $modelsWxs -Encoding UTF8
}

Write-Host "  Done" -ForegroundColor Green

# Step 4: Compile WiX sources
Write-Host "[4/5] Compiling with candle.exe..." -ForegroundColor Yellow

$candleExe = Join-Path $WixPath "candle.exe"
$mainWxs = Join-Path $InstallerDir "CallWhisper.wxs"

# Define source directories
$candleArgs = @(
    "-dSourceDir=$SourceDir",
    "-dVendorDir=$vendorDir",
    "-dModelsDir=$modelsDir",
    "-out", "$ObjDir\",
    "-ext", "WixUIExtension",
    $mainWxs
)

# Only add generated files if they exist and have content
if ((Test-Path $productWxs) -and (Get-Item $productWxs).Length -gt 100) {
    $candleArgs += $productWxs
}
if ((Test-Path $vendorWxs) -and (Get-Item $vendorWxs).Length -gt 100) {
    $candleArgs += $vendorWxs
}
if ((Test-Path $modelsWxs) -and (Get-Item $modelsWxs).Length -gt 100) {
    $candleArgs += $modelsWxs
}

& $candleExe @candleArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Compilation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "  Done" -ForegroundColor Green

# Step 5: Link to create MSI
Write-Host "[5/5] Linking with light.exe..." -ForegroundColor Yellow

$lightExe = Join-Path $WixPath "light.exe"
$msiPath = Join-Path $DistDir "CallWhisper-Setup.msi"

$wixobjFiles = Get-ChildItem -Path $ObjDir -Filter "*.wixobj" | ForEach-Object { $_.FullName }

& $lightExe @wixobjFiles `
    -ext WixUIExtension `
    -out $msiPath `
    -b $SourceDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Linking failed!" -ForegroundColor Red
    exit 1
}

Write-Host "  Done" -ForegroundColor Green

# Calculate package sizes
$msiSize = (Get-Item $msiPath).Length / 1MB

# Get component sizes
$appSize = 0
$vendorSize = 0
$modelSize = 0

Get-ChildItem -Path $SourceDir -Recurse -File | ForEach-Object {
    $relativePath = $_.FullName.Substring($SourceDir.Length + 1)
    $size = $_.Length

    if ($relativePath -like "vendor\*") {
        $vendorSize += $size
    } elseif ($relativePath -like "models\*") {
        $modelSize += $size
    } else {
        $appSize += $size
    }
}

$appSizeMB = $appSize / 1MB
$vendorSizeMB = $vendorSize / 1MB
$modelSizeMB = $modelSize / 1MB
$totalSize = $appSize + $vendorSize + $modelSize
$totalSizeMB = $totalSize / 1MB

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MSI Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "MSI Installer: $msiPath" -ForegroundColor White
Write-Host ""
Write-Host "Package Size Summary:" -ForegroundColor Cyan
Write-Host "  MSI File:     $([math]::Round($msiSize, 1)) MB" -ForegroundColor White
Write-Host ""
Write-Host "  Components:" -ForegroundColor Yellow
Write-Host "    Application:  $([math]::Round($appSizeMB, 1)) MB" -ForegroundColor Gray
Write-Host "    Vendor (FFmpeg, Whisper): $([math]::Round($vendorSizeMB, 1)) MB" -ForegroundColor Gray
Write-Host "    Models (ggml-medium.en): $([math]::Round($modelSizeMB, 1)) MB" -ForegroundColor Gray
Write-Host "    ------------------------------------" -ForegroundColor DarkGray
Write-Host "    Total Installed Size: $([math]::Round($totalSizeMB, 1)) MB" -ForegroundColor White
Write-Host ""
Write-Host "Enterprise Deployment Options:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Silent Install:" -ForegroundColor Yellow
Write-Host "    msiexec /i CallWhisper-Setup.msi /qn" -ForegroundColor Gray
Write-Host ""
Write-Host "  Silent Install with Log:" -ForegroundColor Yellow
Write-Host "    msiexec /i CallWhisper-Setup.msi /qn /l*v install.log" -ForegroundColor Gray
Write-Host ""
Write-Host "  SCCM/GPO:" -ForegroundColor Yellow
Write-Host "    Deploy using standard MSI deployment methods" -ForegroundColor Gray
Write-Host ""
Write-Host "  Uninstall:" -ForegroundColor Yellow
Write-Host "    msiexec /x CallWhisper-Setup.msi /qn" -ForegroundColor Gray
Write-Host ""
Write-Host "Installation Location:" -ForegroundColor Cyan
Write-Host "  C:\Program Files\CallWhisper\" -ForegroundColor Gray
Write-Host ""
Write-Host "Data Location:" -ForegroundColor Cyan
Write-Host "  C:\Program Files\CallWhisper\data\" -ForegroundColor Gray
Write-Host ""
