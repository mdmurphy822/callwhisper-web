# Changelog

All notable changes to CallWhisper will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Multi-language support with automatic language detection
- Speaker diarization (who said what)
- Full-text search across transcripts
- GPU acceleration for whisper.cpp

---

## [1.0.0] - 2024-12-29

### Added

#### Core Features
- **Web-based UI** with real-time WebSocket updates
- **Audio recording** from virtual audio devices (VB-Cable, Stereo Mix)
- **Local transcription** using whisper.cpp (no cloud dependencies)
- **VTB bundle format** - portable ZIP-based container with audio + transcript
- **Multiple export formats** - TXT, SRT, VTT, CSV, PDF, DOCX
- **Batch processing** - upload multiple files or scan folders

#### Security
- **Network Guard** - blocks all external network connections at socket level
- **Device Guard** - prevents recording from microphones (allowlist/blocklist)
- **Rate limiting** - configurable sliding window rate limiter
- **CORS protection** - configurable allowed origins
- **Offline-only mode** - zero external API calls

#### Reliability
- **Crash recovery** - checkpoint system for interrupted transcriptions
- **Bulkhead pattern** - isolated thread pools prevent cascade failures
- **Adaptive timeouts** - dynamic timeout based on audio duration
- **Graceful degradation** - circuit breaker patterns

#### Windows-Specific
- **Automatic call detection** for Cisco Jabber and Finesse
- **WASAPI audio monitoring** for real-time audio session detection
- **Process monitoring** via WMI for call application detection
- **Portable deployment** - single folder, no installation required
- **MSI installer** option for enterprise deployment

#### API
- **47 REST endpoints** for full automation
- **WebSocket API** for real-time updates
- **Swagger/OpenAPI documentation** at `/docs`
- **Health checks** for monitoring and load balancers

#### Developer Experience
- **Comprehensive documentation** - 8 documentation files
- **54 test files** with unit and integration tests
- **Structured logging** with structlog
- **Distributed tracing** with request IDs

### Configuration
- JSON-based configuration in `config/config.json`
- 10 configuration sections:
  - `server` - host, port, browser launch
  - `audio` - sample rate, channels, format
  - `transcription` - model, language, beam size
  - `output` - directory, bundle format
  - `device_guard` - allowlist, blocklist
  - `security` - CORS, rate limiting, debug endpoints
  - `performance` - concurrency, caching
  - `timeouts` - operation timeouts
  - `call_detector` - auto-detection settings

### Architecture
- **FastAPI backend** with async support
- **Vanilla JavaScript frontend** (no framework dependencies)
- **50 Python backend modules** across 4 layers
- **21 core infrastructure modules** implementing enterprise patterns
- **18 service modules** for business logic

### Supported Platforms
- Windows 10/11 (primary target)
- Linux (Ubuntu 20.04+, Fedora 35+)
- macOS (experimental)

### Supported Audio Formats
- Input: WAV, MP3, M4A, FLAC, OGG, OPUS, WMA
- Output: OPUS (recommended), WAV, MP3

### Whisper Models
- ggml-tiny.en.bin (75 MB)
- ggml-base.en.bin (142 MB)
- ggml-small.en.bin (466 MB)
- ggml-medium.en.bin (1.5 GB) - default
- ggml-large-v3.bin (3 GB)

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2024-12-29 | Initial release with full feature set |

---

## Migration Guide

### From Earlier Versions

This is the initial release. No migration required.

### Future Migrations

Migration guides will be added here when breaking changes occur.

---

## Security Advisories

No security advisories for this version.

Report security issues privately to the maintainers.

---

## Contributors

- Initial development by the CallWhisper team

---

## Links

- [Documentation](docs/)
- [API Reference](docs/API.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [GitHub Issues](https://github.com/callwhisper/callwhisper-web/issues)
