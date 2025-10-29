"""
Фабрика для создания провайдеров реранкеров
"""
from .reranker_providers import (
    BaseRerankerProvider,
    OllamaRerankerProvider,
    LocalBGERerankerProvider,
    LocalJinarerankerProvider
)
from src.config.settings import RerankerProviderType

class RerankerProviderFactory:

    @staticmethod
    def create_provider(
        provider_type: RerankerProviderType
    ) -> BaseRerankerProvider:
        
        if provider_type == RerankerProviderType.OLLAMA:
            return OllamaRerankerProvider()
        elif provider_type == RerankerProviderType.LocalBGE:
            return LocalBGERerankerProvider()
        elif provider_type == RerankerProviderType.LocalJina:
            return LocalJinarerankerProvider()
        else:
            raise ValueError(f"Неподдерживаемый тип провайдера: {provider_type}")