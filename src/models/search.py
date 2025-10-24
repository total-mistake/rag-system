from .document import Document
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class SearchResult:
    document: Document
    vector_score: float
    rerank_score: Optional[int | float] = None
    final_score: Optional[float] = None

    def __post_init__(self):
        self.update_final_score()
    
    def update_final_score(self):
        """Пересчитывает final_score на основе доступных скоров"""
        if self.rerank_score is not None:
            self.final_score = self.rerank_score
        else:
            # Используем векторный скор (distance, меньше = лучше)
            self.final_score = max(0, 1 - self.vector_score)
