# Building CallWhisper

This document describes how to build CallWhisper from source for distribution.

## Build Outputs

| Output | Description | Size |
|--------|-------------|------|
| `dist/CallWhisper/` | Portable folder | ~1.6 GB |
| `dist/CallWhisper_Windows_x64.zip` | Portable ZIP | ~1.6 GB |
| `dist/CallWhisper-Setup.msi` | MSI installer | ~1.6 GB |

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Application runtime |
| PowerShell | 5.1+ | Build scripts |
| WiX Toolset | 3.11+ | MSI generation (optional) |

### Installing Prerequisites

**Python:**
```powershell
# Download from python.org or use winget
winget install Python.Python.3.11
```

**WiX Toolset (for MSI builds):**
```powershell
# Download from https://wixtoolset.org/releases/
# Or use Chocolatey
choco install wixtoolset
```

---

## Build Process Overview

```
1. Download vendor binaries (scripts/download-vendor.ps1)
        ↓
2. Build portable app (scripts/build.ps1)
        ↓
3. Build MSI installer (scripts/build-msi.ps1) [optional]
```

---

## Step 1: Download Vendor Binaries

Before building, download the required third-party binaries.

### Automated Download

```powershell
cd callwhisper-web
.\scripts\download-vendor.ps1
```

This downloads:
- **FFmpeg** (~30 MB) - Audio processing
- **whisper-cli.exe** (~5 MB) - Transcription engine
- **ggml-medium.en.bin** (~1.5 GB) - Whisper model

### Manual Download

If the automated script fails, download manually:

| Component | Source | Destination |
|-----------|--------|-------------|
| FFmpeg | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) | `vendor/ffmpeg.exe`, `vendor/ffprobe.exe` |
| Whisper CLI | [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp/releases) | `vendor/whisper-cli.exe` |
| Whisper Model | [HuggingFace](https://huggingface.co/ggerganov/whisper.cpp) | `models/ggml-medium.en.bin` |

### Verify Vendor Files

```powershell
# Should show all files present with sizes
dir vendor\
dir models\
```

Expected structure:
```
vendor/
    ffmpeg.exe      (~100 MB)
    ffprobe.exe     (~100 MB)
    whisper-cli.exe (~5 MB)

models/
    ggml-medium.en.bin (~1.5 GB)
```

---

## Step 2: Build Portable Application

```powershell
.\scripts\build.ps1
```

This will:
1. **Pre-flight checks** - Verify vendor binaries exist (fails if missing)
2. **Clean** - Remove previous build artifacts
3. **Install dependencies** - pip install requirements.txt
4. **PyInstaller** - Bundle Python + dependencies into EXE
5. **Copy vendor files** - Include ffmpeg, whisper-cli, model
6. **Create version.json** - Enable portable mode detection
7. **Create ZIP** - Package for distribution

### Build Output

```
dist/
    CallWhisper/              # Portable folder
        CallWhisper.exe       # Main application
        version.json          # Version info
        vendor/               # FFmpeg, Whisper CLI
        models/               # Whisper model
        data/                 # Config, output, logs
    CallWhisper_Windows_x64.zip
```

### Testing the Build

```powershell
# Test portable build
.\dist\CallWhisper\CallWhisper.exe

# Open browser to http://localhost:8765
```

---

## Step 3: Build MSI Installer (Optional)

Requires WiX Toolset 3.11+.

```powershell
.\scripts\build-msi.ps1
```

This will:
1. Verify portable build exists
2. Create license file if needed
3. Harvest components with WiX heat.exe
4. Compile with WiX candle.exe
5. Link with WiX light.exe
6. Report package sizes

### MSI Output

```
dist/
    CallWhisper-Setup.msi    # Enterprise installer (~1.6 GB)
```

### Testing MSI Installation

```powershell
# Interactive install (for testing)
msiexec /i dist\CallWhisper-Setup.msi

# Silent install
msiexec /i dist\CallWhisper-Setup.msi /qn

# Silent install with log
msiexec /i dist\CallWhisper-Setup.msi /qn /l*v install.log

# Uninstall
msiexec /x dist\CallWhisper-Setup.msi /qn
```

---

## Build Troubleshooting

### Pre-flight Check Failed

```
BUILD FAILED: Missing Required Files
```

**Solution:** Run `.\scripts\download-vendor.ps1` first, or manually download vendor binaries.

### PyInstaller Fails

```
ERROR: PyInstaller failed!
```

**Solutions:**
1. Ensure Python 3.11+ is installed
2. Run `pip install -r requirements.txt`
3. Check for antivirus blocking PyInstaller

### WiX Not Found

```
ERROR: WiX Toolset not found!
```

**Solution:** Install WiX Toolset from https://wixtoolset.org/releases/

### MSI Build Fails

```
ERROR: Compilation failed!
```

**Solutions:**
1. Ensure portable build exists (`.\scripts\build.ps1` first)
2. Check `installer/` directory has CallWhisper.wxs
3. Review WiX error output for specific issues

---

## Development Build (Quick)

For development without creating full distribution:

```powershell
# Install dependencies
pip install -r requirements.txt

# Run directly (requires vendor binaries in place)
python pyinstaller_entrypoint.py

# Or run the source directly
cd src
python -m callwhisper.main
```

---

## Build Configuration

### Changing Whisper Model

To use a different model size:

1. Download desired model from [HuggingFace](https://huggingface.co/ggerganov/whisper.cpp)
2. Place in `models/`
3. Update `config/config.json`:
   ```json
   "transcription": {
       "model": "ggml-base.en.bin"
   }
   ```

Available models:
| Model | Size | RAM | Speed | Quality |
|-------|------|-----|-------|---------|
| tiny.en | 75 MB | ~1 GB | Fastest | Lower |
| base.en | 150 MB | ~1 GB | Fast | Good |
| small.en | 500 MB | ~2 GB | Medium | Better |
| **medium.en** | **1.5 GB** | **~4 GB** | **Slower** | **Best** |

### Changing Build Version

Edit `src/callwhisper/__init__.py`:
```python
__version__ = "1.1.0"
```

---

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Build CallWhisper

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: windows-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install WiX
        run: choco install wixtoolset

      - name: Download vendor binaries
        run: .\scripts\download-vendor.ps1

      - name: Build portable
        run: .\scripts\build.ps1

      - name: Build MSI
        run: .\scripts\build-msi.ps1

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: CallWhisper-${{ github.ref_name }}
          path: |
            dist/CallWhisper_Windows_x64.zip
            dist/CallWhisper-Setup.msi
```

---

## Release Checklist

Before releasing a new version:

- [ ] Update version in `src/callwhisper/__init__.py`
- [ ] Run `.\scripts\download-vendor.ps1` (ensure latest binaries)
- [ ] Run `.\scripts\build.ps1` (verify pre-flight passes)
- [ ] Test portable build manually
- [ ] Run `.\scripts\build-msi.ps1` (verify MSI creates)
- [ ] Test MSI silent install: `msiexec /i dist\CallWhisper-Setup.msi /qn`
- [ ] Test MSI uninstall: `msiexec /x dist\CallWhisper-Setup.msi /qn`
- [ ] Verify first-run setup wizard appears (without VB-Cable)
- [ ] Verify recording works (with VB-Cable)
- [ ] Update CHANGELOG.md
- [ ] Tag release in git

---

## Binary Licenses

| Binary | License | Notes |
|--------|---------|-------|
| FFmpeg | GPL | Static build, all codecs included |
| whisper.cpp | MIT | Open-source speech recognition |
| ggml-medium.en | MIT | Whisper model weights |
| VB-Cable | Donationware | Not bundled, user installs separately |

---

## Support

For build issues:
1. Check this document's troubleshooting section
2. Review build script output carefully
3. Ensure all prerequisites are installed
4. Try a clean build (delete `dist/` and `build/` directories)
