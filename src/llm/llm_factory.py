from .llm_clients import (
    BaseLLMClient,
    OllamaClient,
    GigaClient
)
from src.config.settings import LLMClientType, settings

class LLMClientsFactory:

    @staticmethod
    def create_llm_client(
        llm_client_type: LLMClientType
    ) -> BaseLLMClient:
        if llm_client_type == LLMClientType.OLLAMA:
            return OllamaClient(base_url=settings.ollama_base_url, timeout=settings.ollama_timeout)
        elif llm_client_type == LLMClientType.GIGACHAT:
            return GigaClient(settings.gigachat_credentials)
        else:
            raise ValueError(f"Неподдерживаемый тип провайдера: {llm_client_type}")