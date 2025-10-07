"""
Фабрика для создания провайдеров эмбеддингов
"""
from .embedding_providers import (
    BaseEmbeddingProvider, 
    LocalEmbeddingProvider, 
    OllamaEmbeddingProvider
)
from .config import EmbeddingProviderType


class EmbeddingProviderFactory:
    """Фабрика для создания провайдеров эмбеддингов"""
    
    @staticmethod
    def create_provider(
        provider_type: EmbeddingProviderType,
        model_name: str,
        **kwargs
    ) -> BaseEmbeddingProvider:
        """
        Создает провайдер эмбеддингов указанного типа
        
        Args:
            provider_type: Тип провайдера (LOCAL или OLLAMA)
            model_name: Название модели
            **kwargs: Дополнительные параметры для провайдера
            
        Returns:
            Экземпляр провайдера эмбеддингов
            
        Raises:
            ValueError: Если тип провайдера не поддерживается
        """
        
        if provider_type == EmbeddingProviderType.LOCAL:
            return LocalEmbeddingProvider(model_name)
        
        elif provider_type == EmbeddingProviderType.OLLAMA:
            base_url = kwargs.get('base_url', 'http://localhost:11434')
            return OllamaEmbeddingProvider(model_name, base_url)
        
        else:
            raise ValueError(f"Неподдерживаемый тип провайдера: {provider_type}")