from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.endpoints import router
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.get("app.name", "RAG Chatbot API"),
    version=settings.get("app.version", "1.0.0"),
    description="Internal Knowledge Base RAG Chatbot API",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api", tags=["chat"])


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting RAG Chatbot API...")
    logger.info(f"Environment: {settings.get('app.environment', 'development')}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down RAG Chatbot API...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.get("app.name"),
        "version": settings.get("app.version"),
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint at root level."""
    return {"status": "healthy"}


if __name__ == "__main__":
    host = settings.get("api.host", "0.0.0.0")
    port = settings.get("api.port", 8000)
    reload = settings.get("api.reload", True)

    logger.info(f"Starting server at http://{host}:{port}")

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)
