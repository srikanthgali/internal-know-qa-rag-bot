"""
Run Streamlit UI for RAG chatbot.

Usage:
    python run_streamlit.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


def main():
    """Run Streamlit app."""
    port = settings.get("streamlit.port", 8501)
    app_path = Path("ui/streamlit_app/app.py")

    if not app_path.exists():
        logger.error(f"Streamlit app not found at {app_path}")
        sys.exit(1)

    logger.info(f"Starting Streamlit UI at http://localhost:{port}")

    # Set PYTHONPATH environment variable for Streamlit subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(
        [
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        env=env,
    )


if __name__ == "__main__":
    main()
