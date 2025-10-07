"""
Адаптер для интеграции провайдеров эмбеддингов с ChromaDB
"""
from chromadb.utils.embedding_functions import EmbeddingFunction
from .embedding_providers import BaseEmbeddingProvider
from typing import List
import logging

logger = logging.getLogger(__name__)


class ChromaEmbeddingAdapter(EmbeddingFunction):
    """Адаптер для интеграции провайдеров эмбеддингов с ChromaDB"""
    
    def __init__(self, provider: BaseEmbeddingProvider):
        self.provider = provider
        logger.debug(f"Создан адаптер для провайдера: {type(provider).__name__}")
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Получает эмбеддинги для списка текстов
        
        Args:
            texts: Список текстов для векторизации
            
        Returns:
            Список векторов эмбеддингов
        """
        logger.debug(f"Адаптер обрабатывает {len(texts)} текстов")
        return self.provider.encode(texts)