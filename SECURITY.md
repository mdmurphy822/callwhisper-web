# Security Policy

## Overview

CallWhisper is designed with security as a core principle, particularly for enterprise environments where data privacy is critical.

## Security Features

### 1. Offline-Only Operation

CallWhisper operates **100% offline** with zero external dependencies:

- **Network Guard**: All external network connections are blocked at the socket level
- **Local Processing**: All transcription uses local whisper.cpp - no cloud APIs
- **No Telemetry**: Zero data collection or external reporting

Verify offline mode:
```bash
curl http://localhost:8765/api/health
# Response includes: "mode": "offline", "network_guard": "enabled"
```

### 2. Microphone Protection (Device Guard)

A multi-layer defense system prevents recording from physical microphones:

1. **Explicit Blocklist**: Blocks known input devices (Microphone, Webcam, etc.)
2. **Pattern Matching**: Regex-based detection of dangerous device names
3. **Allowlist Approval**: Only explicitly allowed devices (VB-Cable, Stereo Mix)
4. **Fail-Safe Default**: Unknown devices are BLOCKED by default

### 3. Input Validation

All user inputs are validated:

- **Path Traversal Prevention**: All file paths validated to stay within allowed directories
- **Command Injection Prevention**: Device names and parameters sanitized
- **File Upload Validation**: MIME type checking, size limits (500MB max)
- **Request Validation**: Pydantic models with strict field constraints

### 4. Debug Endpoints

Debug endpoints (`/api/debug/*`) are **disabled by default** in production.

To enable for debugging:
```json
{
  "security": {
    "debug_endpoints_enabled": true
  }
}
```

**Warning**: Never enable debug endpoints in production as they expose internal state.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability:

### For Critical Issues (RCE, Auth Bypass, Data Exposure)

1. **Do NOT open a public GitHub issue**
2. Email security concerns to the maintainers directly
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Any suggested fixes

### For Non-Critical Issues

Open a GitHub issue with the `security` label.

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: Within 72 hours
  - High: Within 2 weeks
  - Medium: Within 30 days
  - Low: Next release cycle

## Security Best Practices for Deployment

### 1. Configuration

```json
{
  "security": {
    "debug_endpoints_enabled": false,
    "cors_enabled": true,
    "allowed_origins": ["http://localhost:8765"],
    "rate_limit_enabled": true
  }
}
```

### 2. Binary Verification

Before deployment, verify binary checksums:
```bash
cd vendor/
sha256sum -c checksums.txt
```

See `vendor/checksums.json` for expected values.

### 3. Network Isolation

CallWhisper binds to `127.0.0.1:8765` by default (localhost only).

**Never expose to external networks** without proper authentication.

### 4. File Permissions

```bash
# Recommended permissions
chmod 755 CallWhisper  # or CallWhisper.exe
chmod 755 vendor/ffmpeg
chmod 755 vendor/whisper-cli
chmod 644 config/config.json
chmod 700 output/  # Recording output directory
```

### 5. Dependency Updates

Regularly update dependencies and run security scans:
```bash
pip install safety pip-audit
safety check
pip-audit
```

## Known Security Considerations

### 1. Local Access

Anyone with local access to the machine can use CallWhisper. Consider:
- User session isolation
- File permission hardening
- Audit logging for compliance

### 2. Audio Data

Recordings are stored locally in `output/` directory:
- Implement data retention policies
- Encrypt sensitive recordings
- Secure backup procedures

### 3. Third-Party Binaries

CallWhisper depends on:
- **FFmpeg**: Audio processing
- **whisper.cpp**: Transcription engine

Both are vendored or system-installed. Verify sources before deployment.

## Compliance

CallWhisper assists with compliance requirements:

- **GDPR**: Local processing, no cloud transfer
- **HIPAA**: Supports offline operation for PHI
- **SOC 2**: Audit logging, access controls
- **PCI DSS**: No cardholder data processing

**Note**: Application security is one component. Full compliance requires organizational policies.

## Security Contacts

- **Project Maintainers**: See GitHub repository
- **Security Issues**: Create private security advisory on GitHub

## Changelog

| Date       | Change                                    |
|------------|-------------------------------------------|
| 2026-01-05 | Added debug endpoint controls             |
| 2026-01-05 | Added WebSocket connection logging        |
| 2026-01-05 | Added dependency security scanning        |
| 2026-01-05 | Created initial security documentation    |
