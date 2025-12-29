import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()


@dataclass
class OpenAIConfig:
    """OpenAI configuration."""

    model: str
    embedding_model: str
    temperature: float
    max_tokens: int
    api_key: str


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""

    type: str
    dimension: int
    index_path: Path
    metadata_path: Path


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""

    top_k: int
    similarity_threshold: float
    rerank: bool
    edge_case_min_score: float = 0.80


@dataclass
class GenerationConfig:
    """Generation configuration."""

    system_prompt: str
    include_sources: bool
    max_context_length: int


@dataclass
class DocumentProcessingConfig:
    """Document processing configuration."""

    chunk_size: int
    chunk_overlap: int
    separators: list
    supported_formats: list


class Settings:
    """Application settings manager."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize settings from config file.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        if self._config is not None:
            self._validate_config()

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            print(f"Warning: Config file not found: {self.config_path}")
            print("Using default configuration...")
            return self._get_default_config()

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                if config is None:
                    print(f"Warning: Config file is empty: {self.config_path}")
                    return self._get_default_config()
                return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration...")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "openai": {
                "model": "gpt-4",
                "embedding_model": "text-embedding-3-small",
                "temperature": 0.7,
                "max_tokens": 2000,
                "api_key_env": "OPENAI_API_KEY",
            },
            "vector_store": {
                "type": "faiss",
                "dimension": 1536,
                "index_path": "data/vector_store/faiss_index.bin",
                "metadata_path": "data/vector_store/metadata.json",
            },
            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.7,
                "rerank": False,
            },
            "document_processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", ". ", " ", ""],
                "supported_formats": [".pdf", ".txt", ".docx", ".md"],
            },
            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.75,
                "rerank": False,
                "edge_case_min_score": 0.80,
            },
        }

    def _validate_config(self):
        """Validate required configuration values."""
        if self._config is None:
            print("Warning: Configuration is None, skipping validation")
            return

        # Check for OpenAI API key
        try:
            api_key_env = self._config.get("openai", {}).get(
                "api_key_env", "OPENAI_API_KEY"
            )
            api_key = os.getenv(api_key_env)
            if not api_key:
                print(
                    f"Warning: OpenAI API key not found. Set {api_key_env} "
                    "environment variable or update .env file"
                )
        except Exception as e:
            print(f"Warning: Error validating config: {e}")

    @property
    def openai(self) -> OpenAIConfig:
        """Get OpenAI configuration."""
        if self._config is None:
            self._config = self._get_default_config()

        config = self._config.get("openai", {})
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")

        return OpenAIConfig(
            model=config.get("model", "gpt-4"),
            embedding_model=config.get("embedding_model", "text-embedding-3-small"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            api_key=os.getenv(api_key_env, ""),
        )

    @property
    def vector_store(self) -> VectorStoreConfig:
        """Get vector store configuration."""
        if self._config is None:
            self._config = self._get_default_config()

        config = self._config.get("vector_store", {})
        return VectorStoreConfig(
            type=config.get("type", "faiss"),
            dimension=config.get("dimension", 1536),
            index_path=Path(
                config.get("index_path", "data/vector_store/faiss_index.bin")
            ),
            metadata_path=Path(
                config.get("metadata_path", "data/vector_store/metadata.json")
            ),
        )

    @property
    def retrieval(self) -> RetrievalConfig:
        """Get retrieval configuration."""
        if self._config is None:
            self._config = self._get_default_config()

        config = self._config.get("retrieval", {})
        return RetrievalConfig(
            top_k=config.get("top_k", 5),
            similarity_threshold=config.get("similarity_threshold", 0.7),
            rerank=config.get("rerank", False),
            edge_case_min_score=config.get("edge_case_min_score", 0.80),
        )

    @property
    def generation(self) -> GenerationConfig:
        """Get generation configuration."""
        if self._config is None:
            self._config = self._get_default_config()

        config = self._config.get("generation", {})
        return GenerationConfig(
            system_prompt=config.get(
                "system_prompt",
                "You are a helpful assistant that provides accurate answers based on the provided context.",
            ),
            include_sources=config.get("include_sources", True),
            max_context_length=config.get("max_context_length", 5000),
        )

    @property
    def document_processing(self) -> DocumentProcessingConfig:
        """Get document processing configuration."""
        if self._config is None:
            self._config = self._get_default_config()

        config = self._config.get("document_processing", {})
        return DocumentProcessingConfig(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            separators=config.get("separators", ["\n\n", "\n", ". ", " ", ""]),
            supported_formats=config.get(
                "supported_formats", [".pdf", ".txt", ".docx", ".md"]
            ),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'app.name')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value


# Global settings instance
settings = Settings()
