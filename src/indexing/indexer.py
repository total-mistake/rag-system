from .chroma_manager import ChromaDBManager
from .chroma_embedding_adapter import ChromaEmbeddingAdapter
from ..models.document import DocumentCollection
from .config import EmbeddingProviderType
from .embedding_factory import EmbeddingProviderFactory
import logging
import os

logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self, 
                 embedding_provider: EmbeddingProviderType = EmbeddingProviderType.LOCAL,
                 embedding_model: str = "jinaai/jina-embeddings-v3",
                 chroma_db_path: str = "storage/chroma_db",
                 collection_name: str = "documents",
                 **provider_kwargs):
        
        # Создаем провайдер эмбеддингов
        self.embedding_provider = EmbeddingProviderFactory.create_provider(
            embedding_provider, 
            embedding_model,
            **provider_kwargs
        )
        
        # Создаем адаптер для ChromaDB
        embedding_fn = ChromaEmbeddingAdapter(self.embedding_provider)

        # Инициализируем ChromaDB
        self.db_connector = ChromaDBManager(
            chroma_db_path,
            collection_name,
            embedding_model,
            embedding_fn
        )
        
    def index_documents(self, documents_json_path: str, force_reindex: bool = False):
        try:
            if not os.path.exists(documents_json_path):
                raise FileNotFoundError(f"Файл {documents_json_path} не найден")

            # Проверяем, есть ли уже документы в коллекции
            collection_info = self.db_connector.get_collection_info()
            if collection_info["count"] > 0 and not force_reindex:
                print("Индексация уже существует. Используйте force_reindex=True для переиндексации.")
                return
            
            if force_reindex:
                self.clear_index()

            documents = DocumentCollection.from_json_file(documents_json_path)

            self.db_connector.add_documents(documents)

            logger.info(f"Индексация завершена. Добавлено {len(documents.documents)} документов.")
        except Exception as e:
            logger.error(f"Ошибка при индексации документов: {e}")
            raise
        
    def reindex_with_new_model(self, new_model: str):
        # Переиндексация с новой моделью
        pass
        
    def clear_index(self):
        """Очищает существующий индекс"""
        try:
            collection_name = self.db_connector.collection.name
            embedding_function = self.db_connector.collection._embedding_function
            
            # Удаляем коллекцию
            self.db_connector.client.delete_collection(collection_name)
            
            # Пересоздаем пустую коллекцию
            self.db_connector.collection = self.db_connector.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            
            logger.info("Индекс очищен и пересоздан")
        except Exception as e:
            logger.error(f"Ошибка при очистке индекса: {e}")
            raise