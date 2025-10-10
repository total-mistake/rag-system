import enum
from dataclasses import dataclass

class EmbeddingProviderType(enum.Enum):
    LOCAL = "local"
    OLLAMA = "ollama"

class IndexingConfig:
    # Общие настройки
    chroma_db_path: str = "storage/chroma_db"
    collection_name: str = "documents-test"
    batch_size: int = 1
    show_progress: bool = True

    # Настройки эмбеддингов
    embedding_provider: EmbeddingProviderType = EmbeddingProviderType.OLLAMA
    embedding_model: str = "qwen3-embedding:0.6b"
    # embedding_model: str = "google/embeddinggemma-300m"
    # embedding_model: str = "jinaai/jina-embeddings-v3"

    # Настройки для Ollama
    ollama_base_url: str = "http://172.16.100.164:11434"
