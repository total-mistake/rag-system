"""
Фабрика для создания провайдеров реранкеров
"""
from .reranker_providers import (
    BaseRerankerProvider,
    OllamaRerankerProvider
)
from src.config.settings import RerankerProviderType

class RerankerProviderFactory:

    @staticmethod
    def create_provider(
        provider_type: RerankerProviderType
    ) -> BaseRerankerProvider:
        
        if provider_type == RerankerProviderType.OLLAMA:
            return OllamaRerankerProvider()
        
        else:
            raise ValueError(f"Неподдерживаемый тип провайдера: {provider_type}")