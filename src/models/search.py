from .document import Document
from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchResult:
    document: Document
    vector_score: float
    rerank_score: Optional[int] = None
    final_score: Optional[float] = None

    def __post_init__(self):
        self.update_final_score()
    
    def update_final_score(self):
        """Пересчитывает final_score на основе доступных скоров"""
        if self.rerank_score is not None:
            # Нормализуем скор реранкера (1-5) к диапазону (0-1)
            self.final_score = (self.rerank_score - 1) / 4
        else:
            # Используем векторный скор (distance, меньше = лучше)
            self.final_score = max(0, 1 - self.vector_score)

@dataclass
class SearchMetrics:
    vector_search_time: float
    rerank_time: Optional[float]
    total_time: float

@dataclass
class RerankerResponse:
    score: float
    reasoning: Optional[str] = None