#!/usr/bin/env python
"""
PyInstaller entry point for CallWhisper

This script is the main entry point when running the packaged application.
It handles path resolution for bundled resources and starts the server.
"""

import sys
import os
from pathlib import Path


def setup_environment():
    """Configure environment for PyInstaller bundle."""
    # Determine if running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running as bundle - _MEIPASS contains extracted files
        bundle_dir = Path(sys._MEIPASS)
        data_dir = Path(sys.executable).parent
    else:
        # Running in development
        bundle_dir = Path(__file__).parent
        data_dir = bundle_dir

    # Set environment variables for path resolution
    os.environ['CALLWHISPER_BUNDLE_DIR'] = str(bundle_dir)
    os.environ['CALLWHISPER_DATA_DIR'] = str(data_dir)

    # Ensure we can import the package
    src_path = bundle_dir / 'src'
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main():
    """Main entry point."""
    setup_environment()

    # Import after environment setup
    import uvicorn
    from callwhisper.main import app
    from callwhisper.core.config import get_settings

    settings = get_settings()

    # Open browser if configured
    if settings.server.open_browser:
        import webbrowser
        import threading

        def open_browser():
            import time
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open(f"http://{settings.server.host}:{settings.server.port}")

        threading.Thread(target=open_browser, daemon=True).start()

    print(f"Starting CallWhisper on http://{settings.server.host}:{settings.server.port}")

    # Run server
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
