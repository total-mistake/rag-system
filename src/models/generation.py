from dataclasses import dataclass
from typing import List
from .document import Document

@dataclass
class GenerationResult:
    """Результат генерации"""
    answer: str
    source_urls: List[str]
    source_documents: List[Document]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_duration: int
    load_duration: int
    eval_duration: int
    model_used: str