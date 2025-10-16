from ..models.search import SearchResult
from ..models.ollama_client import OllamaClient
from ..models.pipeline import RerankingResult, StageMetrics
from src.config.settings import settings
from typing import List
from ..config.prompts import *
import time
import logging
import re

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self):
        self.client  = OllamaClient(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout
        )
        self.timeout = settings.ollama_timeout

        # Параметры для реранкера из настроек
        self.model = settings.reranker_model
        self.temperature = settings.reranker_temperature
        self.top_p = settings.reranker_top_p
        self.top_k = settings.reranker_top_k

    def rerank(self, query: str, documents: List[SearchResult]) -> RerankingResult:
        if not self.client.is_healthy():
            logger.info("Этап реранжирования пропущен из-за сбоя доступа к серверу Ollama")
            return documents
        
        start_time = time.time()
        total_tokens = 0

        for doc in documents:
            try:
                content = f"{doc.document.title}\n{doc.document.text}"
                
                response = self.client.chat(
                    model=self.model,
                    messages=[RERANK_SYSTEM_PROMPT,create_rerank_user_prompt(query, content)],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k
                )

                score = self._extract_score(response.content)
                doc.rerank_score = score
                doc.update_final_score()

                total_tokens += response.prompt_eval_count + response.eval_count

                logger.info(f"Реранкинг: {response.prompt_eval_count} входных + {response.eval_count} выходных токенов.\nВремя загрузки: {response.load_duration}\nВремя генерации ответа: {response.eval_duration}\nОбщее время: {response.total_duration}")
            except Exception as e:
                logger.error(f"Ошибка при реранжировании документа: {e}")
        
        total_time = time.time() - start_time
        result = RerankingResult(
            StageMetrics("rerank", total_time),
            documents,
            total_tokens
        )
        return result


    def _extract_score(self, content: str) -> int:
        """Извлечение оценки из ответа модели"""

        numbers = re.findall(r'\b[1-5]\b', content)
        
        if not numbers:
            raise ValueError(f"Не удалось извлечь корректную оценку из ответа: '{content}'")
        
        return float(numbers[0])