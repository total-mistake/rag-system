from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum

class EmbeddingProviderType(str, Enum):
    LOCAL = "local"
    OLLAMA = "ollama"

class RerankerProviderType(str, Enum):
    OLLAMA = "ollama"
    LocalBGE = "bge"
    LocalJina = "jina"

class LLMClientType(str, Enum):
    OLLAMA = "ollama"
    GIGACHAT = "gigachat"

class AppSettings(BaseSettings):
    # Общие настройки приложения
    app_name: str = Field(default="RAG System", description="Название приложения")
    
    # Настройки ChromaDB
    chroma_db_path: str = Field(default="storage/chroma_db", description="Путь к ChromaDB")
    collection_name: str = Field(default="documents", description="Имя коллекции")
    
    # Настройки эмбеддингов
    # text_splitter: bool = Field(default=False, description="Использовать сегментацию текста для запроса пользователя")
    embedding_provider: EmbeddingProviderType = Field(default=EmbeddingProviderType.OLLAMA)
    embedding_model: str = Field(default="qwen3-embedding:0.6b", description="Модель эмбеддингов")
    batch_size: int = Field(default=1, ge=1, description="Размер батча для индексации")
    show_progress: bool = Field(default=True, description="Показывать прогресс")
    
    # Настройки LLM
    ollama_base_url: str = Field(default="http://172.16.100.164:11434", description="URL Ollama сервера")
    ollama_timeout: int = Field(default=30, ge=1, description="Таймаут запросов к Ollama")
    max_concurrent_requests: int = Field(default=5)
    llm_client: LLMClientType = Field(default=LLMClientType.OLLAMA)
    gigachat_credentials: str
    
    # Настройки поиска
    initial_candidates: int = Field(default=10, ge=1, description="Кандидаты из векторного поиска")
    final_results: int = Field(default=5, ge=1, description="Финальные результаты")
    max_context_documents: int = Field(default=5, ge=1, le=20, description="Максимум документов в контексте")
    enable_hyde: bool = Field(default=False, description="Использовать в ретривере подход HyDE для улучшения векторного поиска")
    
    # Настройки реранкера
    reranker_provider: RerankerProviderType = Field(default=RerankerProviderType.OLLAMA)
    enable_reranking: bool = Field(default=True, description="Включить реранкинг")
    reranker_temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Температура модели реранкера")
    reranker_model: str = Field(default="gemma3:4b-it-qat", description="Модель реранкера")
    reranker_top_k: int = Field(default=40, ge=1, description="Топ-к реранкера")
    reranker_top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Топ-п реранкера")
    document_filtering_threshold: int = Field(default=0, description="Порог, результаты с оценкой меньше которых отбрасываются после реранкинга")
    
    # Настройки генерации ответа
    generation_model: str = Field(default="gemma3:4b-it-qat", description="Модель генерации ответа")
    generation_temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Температура модели генерации ответа")
    generation_top_k: int = Field(default=40, ge=1, description="Топ-к генерации ответа")
    generation_top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Топ-п генерации ответа")

    # Настройки телеграм бота
    telegram_token: str = Field(default=None, description="Токен аутентификации бота")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Создаем глобальный экземпляр настроек
settings = AppSettings()