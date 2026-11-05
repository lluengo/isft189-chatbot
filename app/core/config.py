from functools import lru_cache
from typing import Optional
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración global de la aplicación gestionada mediante variables de entorno.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # General
    PROJECT_NAME: str = "Chatbot Institucional API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True

    # Base de Datos PostgreSQL
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "Holamundo123"
    POSTGRES_DB: str = "chatbot_db"

    @computed_field
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Proveedor y Modelo de Embeddings
    EMBEDDING_PROVIDER: str = "openai"  # Opciones: "openai", "mock"
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # Parámetros de Segmentación (RAG Chunking)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Proveedor y Modelo de Lenguaje (LLM) para Chat y Tool Calling
    LLM_PROVIDER: str = "openai"  # Opciones: "openai", "ollama", "mock"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # Configuración de Ollama (Local)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "llama3:8b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
