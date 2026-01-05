#!/bin/bash
# CallWhisper Development Run Script (Linux/macOS)
# Runs the application in development mode

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Starting CallWhisper in development mode..."
echo ""

cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Add src to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src"

# Run the application
echo "Starting server..."
echo ""

python -m uvicorn callwhisper.main:app --reload --host 127.0.0.1 --port 8765
