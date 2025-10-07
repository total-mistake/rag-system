"""
Провайдеры эмбеддингов для различных источников
"""
from abc import ABC, abstractmethod
from typing import List
import requests
import json
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Абстрактный базовый класс для провайдеров эмбеддингов"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Получить эмбеддинги для списка текстов"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Получить размерность векторов"""
        pass


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Локальная модель через SentenceTransformers"""
    
    def __init__(self, model_name: str):
        logger.info(f"Инициализация локальной модели эмбеддингов: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        logger.debug(f"Векторизация {len(texts)} текстов через локальную модель")
        return self.model.encode(
            texts, 
            task='retrieval.passage',
            batch_size=1,
            show_progress_bar=True
        ).tolist()
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Эмбеддинги через API Ollama"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._dimension = None
        logger.info(f"Инициализация Ollama провайдера: {model_name} на {base_url}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        logger.debug(f"Векторизация {len(texts)} текстов через Ollama API")
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                logger.debug(f"Обрабатываем текст {i+1}/{len(texts)}")
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=30
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                embeddings.append(embedding)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка при получении эмбеддинга для текста {i+1}: {e}")
                raise
            except KeyError as e:
                logger.error(f"Неверная структура ответа от Ollama API: {e}")
                raise
                
        return embeddings
    
    def get_dimension(self) -> int:
        if self._dimension is None:
            # Получаем размерность, сделав тестовый запрос
            logger.debug("Определяем размерность эмбеддингов")
            test_embedding = self.encode(["test"])[0]
            self._dimension = len(test_embedding)
            logger.info(f"Размерность эмбеддингов: {self._dimension}")
        return self._dimension