
from src.indexing.indexer import DocumentIndexer
from src.indexing.config import IndexingConfig, EmbeddingProviderType
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Ollama API
indexer_ollama = DocumentIndexer(
    embedding_provider=EmbeddingProviderType.OLLAMA,
    embedding_model=IndexingConfig.ollama_model,
    chroma_db_path=IndexingConfig.chroma_db_path,
    collection_name=IndexingConfig.collection_name,
    base_url=IndexingConfig.ollama_base_url
)

print("Начинаем индексацию документов через Ollama...")
indexer_ollama.index_documents("storage/documents/documents.json")