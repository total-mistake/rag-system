
from src.indexing.indexer import DocumentIndexer
from src.config import settings
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Ollama API
indexer_ollama = DocumentIndexer(
    embedding_provider=settings.embedding_provider,
    embedding_model=settings.embedding_model,
    chroma_db_path=settings.chroma_db_path,
    collection_name=settings.collection_name
)

print("Начинаем индексацию документов через Ollama...")
indexer_ollama.index_documents("storage/documents/documents.json")