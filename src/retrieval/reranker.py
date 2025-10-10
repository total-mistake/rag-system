from ..models.search import SearchResult, RerankerResponse
from typing import List
from .prompts import *
import requests
import logging
import re

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self, model_name: str, base_url: str, timeout: int = 30):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout

    def rerank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        if self._check_ollama_health():
            for doc in documents:
                try:
                    content = f"{doc.document.title}\n{doc.document.text}"
                    response = self._call_ollama_rerank(query, content)
                    doc.rerank_score = int(response.score)
                    doc.update_final_score()
                except Exception as e:
                    logger.error(f"Ошибка при реранжировании документа: {e}")
        else:
            logger.info("Этап реранжирования пропущен из-за сбоя доступа к серверу Ollama")
        
        return documents


    def _check_ollama_health(self):
        try:
            response = requests.get(self.base_url)
            if 200 <= response.status_code < 300:
                logger.info(f"Сервер Ollama по адресу {self.base_url} доступен")
                return True
            else:
                logger.error(f"Сервер Ollama по адресу {self.base_url} недоступен")
                return False
        except requests.exceptions.RequestException as e:
            logger.exception(f"Произошла ошибка при запросе к {self.base_url}: {e}")
            return False


    def _call_ollama_rerank(self, query: str, document_text: str) -> RerankerResponse:
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": create_rerank_user_prompt(query, document_text)}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()
            content = result.get("message", {}).get("content", "").strip()
            
            numbers = re.findall(r'\b[1-5]\b', content)
        
            if not numbers:
                raise ValueError(f"Не удалось извлечь корректную оценку из ответа: '{content}'")
            
            score = float(numbers[0])
            return RerankerResponse(score=score, reasoning=content)
    
        except Exception as e:
            logger.error(f"Ошибка при вызове Ollama: {e}")
            raise
