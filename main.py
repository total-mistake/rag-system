
from src.indexing.indexer import DocumentIndexer
from src.indexing.config import IndexingConfig
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Ollama API
indexer_ollama = DocumentIndexer(
    embedding_provider=IndexingConfig.embedding_provider,
    embedding_model=IndexingConfig.embedding_model,
    chroma_db_path=IndexingConfig.chroma_db_path,
    collection_name=IndexingConfig.collection_name
)

print("Начинаем индексацию документов через Ollama...")
indexer_ollama.index_documents("storage/documents/documents.json")