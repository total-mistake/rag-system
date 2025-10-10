from dataclasses import dataclass

@dataclass
class RetrieverConfig:
    # Параметры векторного поиска
    initial_candidates: int = 10  # Количество кандидатов из векторного поиска
    final_results: int = 5        # Количество финальных результатов
    
    # Параметры реранкера
    enable_reranking: bool = True
    reranker_model: str = "gemma3:4b-it-qat"
    reranker_timeout: int = 30
    
    # Ollama настройки
    ollama_base_url: str = "http://172.16.100.164:11434"