#!/bin/bash
# Local CI script - runs the same checks as GitHub Actions
# Usage: ./scripts/ci-local.sh          # Quick: lint + unit tests (~25s)
#        ./scripts/ci-local.sh --full   # Full: all tests + security (~5m)

set -e

cd "$(dirname "$0")/.."

FULL_MODE=false
if [ "$1" == "--full" ]; then
    FULL_MODE=true
fi

if [ "$FULL_MODE" = true ]; then
    echo "=== Local CI Check (FULL) ==="
else
    echo "=== Local CI Check (QUICK) ==="
fi
echo ""

# 1. Lint - Black
echo "[1/4] Checking formatting with Black..."
if .venv/bin/black --check --diff src/; then
    echo "  Black: PASSED"
else
    echo "  Black: FAILED - run 'black src/' to fix"
    exit 1
fi
echo ""

# 2. Lint - Ruff
echo "[2/4] Linting with Ruff..."
if .venv/bin/ruff check src/; then
    echo "  Ruff: PASSED"
else
    echo "  Ruff: FAILED"
    exit 1
fi
echo ""

# 3. Tests
if [ "$FULL_MODE" = true ]; then
    echo "[3/4] Running ALL tests..."
    if .venv/bin/python -m pytest --cov=src/callwhisper -v --timeout=60; then
        echo "  Tests: PASSED"
    else
        echo "  Tests: FAILED"
        exit 1
    fi
else
    # Skip slow orchestrator tests in quick mode (they have threading issues with signal-based timeout)
    echo "[3/4] Running UNIT tests only (use --full for all tests)..."
    if .venv/bin/python -m pytest tests/unit/ \
        --ignore=tests/unit/services/test_process_orchestrator.py \
        --ignore=tests/unit/services/test_transcriber.py \
        --cov=src/callwhisper -q --timeout=30; then
        echo "  Unit Tests: PASSED (skipped slow tests)"
    else
        echo "  Unit Tests: FAILED"
        exit 1
    fi
fi
echo ""

# 4. Security (only in full mode)
if [ "$FULL_MODE" = true ]; then
    echo "[4/4] Running security scans..."

    echo "  Running Bandit..."
    .venv/bin/bandit -r src/ -ll || true

    echo "  Running pip-audit..."
    .venv/bin/pip-audit || true

    echo "  Security scans: DONE"
    echo ""
else
    echo "[4/4] Skipping security scans (use --full to include)"
    echo ""
fi

echo "=== All checks passed ==="
