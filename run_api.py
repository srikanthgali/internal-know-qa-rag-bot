"""
Run FastAPI server for RAG chatbot.

Usage:
    python run_api.py
"""

import uvicorn
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


def main():
    """Run FastAPI server."""
    host = settings.get("api.host", "0.0.0.0")
    port = settings.get("api.port", 8000)
    reload = settings.get("api.reload", True)

    logger.info(f"Starting FastAPI server at http://{host}:{port}")
    logger.info("API Documentation: http://localhost:8000/docs")

    uvicorn.run("api.main:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    main()
