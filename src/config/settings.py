from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum
from typing import Optional

class EmbeddingProviderType(str, Enum):
    LOCAL = "local"
    OLLAMA = "ollama"

class AppSettings(BaseSettings):
    # Общие настройки приложения
    app_name: str = Field(default="RAG System", description="Название приложения")
    
    # Настройки ChromaDB
    chroma_db_path: str = Field(default="storage/chroma_db", description="Путь к ChromaDB")
    collection_name: str = Field(default="documents", description="Имя коллекции")
    
    # Настройки эмбеддингов
    embedding_provider: EmbeddingProviderType = Field(default=EmbeddingProviderType.OLLAMA)
    embedding_model: str = Field(default="qwen3-embedding:0.6b", description="Модель эмбеддингов")
    batch_size: int = Field(default=1, ge=1, description="Размер батча для индексации")
    show_progress: bool = Field(default=True, description="Показывать прогресс")
    
    # Настройки Ollama
    ollama_base_url: str = Field(default="http://172.16.100.164:11434", description="URL Ollama сервера")
    ollama_timeout: int = Field(default=30, ge=1, description="Таймаут запросов к Ollama")
    
    # Настройки поиска
    initial_candidates: int = Field(default=10, ge=1, description="Кандидаты из векторного поиска")
    final_results: int = Field(default=5, ge=1, description="Финальные результаты")
    
    # Настройки реранкера
    enable_reranking: bool = Field(default=True, description="Включить реранкинг")
    reranker_model: str = Field(default="gemma3:4b-it-qat", description="Модель реранкера")
    reranker_timeout: int = Field(default=30, ge=1, description="Таймаут реранкера")
    

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Позволяет переопределять настройки через переменные окружения
        env_prefix = "RAG_"  # Префикс для переменных окружения

# Создаем глобальный экземпляр настроек
settings = AppSettings()