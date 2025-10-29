from dataclasses import dataclass, field
from typing import List, Optional
from .search import SearchResult
from .document import Document

@dataclass
class StageMetrics:
    """Базовые метрики для любого этапа"""
    stage_name: str
    duration: Optional[float] = None    # В секундах

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
class GeneralResults:
    total_tokens: int = None
    total_duration: float = None

@dataclass
class RAGPipeline:
    """Полная информация о выполнении RAG пайплайна"""
    query: str
    retriever: Optional[VectorSearchResult] = None
    reranking: Optional[RerankingResult] = None
    generation: Optional[GenerationResult] = None
    general: Optional[GeneralResults] = None

    def update_general_results(self):
        self.general = GeneralResults()
        self.general.total_duration = self.retriever.metrics.duration + (self.retriever.metrics.duration or 0) + self.generation.metrics.duration
        self.general.total_tokens = (self.reranking.total_tokens_used or 0) + self.generation.total_tokens
