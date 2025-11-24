from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

#Configuration management for the RAG application.


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/rag_db"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    vector_collection_name: str = "documents"
    
    # API Keys
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # all-MiniLM-L6-v2 dimension
    
    # LLM Configuration
    llm_provider: str = "groq"  # "ollama", "groq", or "openai"
    llm_model: str = "llama-3.1-8b-instant"  # ollama: llama3.2, groq: llama-3.1-8b-instant, openai: gpt-3.5-turbo
    
    # Application
    app_name: str = "RAG Assessment API"
    debug: bool = False
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # This allows extra fields from .env
    )


# Global settings instance
settings = Settings()