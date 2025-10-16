from dataclasses import dataclass, field
from typing import List, Optional
from .search import SearchResult
from .document import Document

@dataclass
class StageMetrics:
    """Базовые метрики для любого этапа"""
    stage_name: str
    duration: Optional[float] = None    # В секундах
    # success: bool = False
    # error_message: Optional[str] = None

@dataclass
class VectorSearchResult:
    """Результат векторного поиска"""
    metrics:  StageMetrics
    results: List[SearchResult] = field(default_factory=list)

@dataclass
class RerankingResult:
    """Результат реранжирования"""
    metrics: StageMetrics
    results: List[SearchResult] = field(default_factory=list)
    total_tokens_used: Optional[int] = None

@dataclass
class GenerationResult:
    """Результат генерации ответа"""
    metrics: StageMetrics
    answer: str
    source_urls: List[str]
    source_documents: List[Document]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    load_duration: int
    eval_duration: int
    model_used: str

@dataclass
class RAGPipeline:
    """Полная информация о выполнении RAG пайплайна"""
    query: str
    vector_search: Optional[VectorSearchResult] = None
    reranking: Optional[RerankingResult] = None
    generation: Optional[GenerationResult] = None
    total_duration: Optional[float] = None

    def get_total_duration(self) -> float:
        """Вычисление общего времени выполнения"""
        durations = []
        if self.vector_search and self.vector_search.metrics.duration:
            durations.append(self.vector_search.metrics.duration)
        if self.reranking and self.reranking.metrics.duration:
            durations.append(self.reranking.metrics.duration)
        if self.generation and self.generation.metrics.duration:
            durations.append(self.generation.metrics.duration)
        
        return sum(durations)
