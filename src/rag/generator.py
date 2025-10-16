from ..models.ollama_client import OllamaClient
from ..models.pipeline import GenerationResult, StageMetrics
from ..models.document import Document
from ..config.prompts import RAG_SYSTEM_PROMPT, create_rag_user_prompt, format_document_context
from ..config.settings import settings

from typing import List
import logging
import re
import time

class ResponseGenerator:
    """Генератор ответов на основе найденных документов"""

    def __init__(self):
        self.client = OllamaClient(
            base_url=settings.ollama_base_url,
            timeout=300
        )
        self.logger = logging.getLogger(__name__)

        self.model = settings.generation_model
        self.temperature = settings.generation_temperature
        self.top_p = settings.generation_top_p
        self.top_k = settings.generation_top_k

    def generate_answer(self,
                        query: str,
                        documents: List[Document]
    ) -> GenerationResult:
        """Генерирует ответ на основе найденных документов"""

        try:
            start_time = time.time()
            selected_docs = documents[:settings.max_context_documents]
            context = format_document_context(selected_docs)
            
            messages = [RAG_SYSTEM_PROMPT, create_rag_user_prompt(query, context)]

            response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k
            )

            self.logger.info(f"Сгенерированный ответ:\n {response.content}")

            # Парсинг ответа и извлечение URL
            answer, source_urls = self._parse_response(response.content)

            self.logger.info(f"Генерация ответа: {response.prompt_eval_count} входных + {response.eval_count} выходных токенов.\nВремя загрузки: {response.load_duration}\nВремя генерации ответа: {response.eval_duration}\nОбщее время: {response.total_duration}")

            return GenerationResult(
                metrics=StageMetrics("generation", time.time() - start_time),
                answer=answer,
                source_urls=source_urls,
                source_documents=selected_docs,
                input_tokens=response.prompt_eval_count,
                output_tokens=response.eval_count,
                total_tokens=response.prompt_eval_count + response.eval_count,
                load_duration = response.load_duration / 1_000_000_000,
                eval_duration = response.eval_duration / 1_000_000_000,
                model_used=response.model
            )

        except Exception as e:
            self.logger.error(f"Ошибка при генерации ответа: {e}")
            raise

    def _parse_response(self, content: str) -> tuple[str, List[str]]:
        """Парсинг ответа модели для извлечения текста и URL"""
        
        # Разделяем ответ на основную часть и источники
        parts = content.split("Источники:")
        
        if len(parts) == 2:
            answer = parts[0].strip()
            sources_text = parts[1].strip()
            
            # Извлекаем URL из источников
            urls = self._extract_urls(sources_text)
        else:
            answer = content.strip()
            urls = []
        
        return answer, urls
    
    def _extract_urls(self, text: str) -> List[str]:
        """Извлечение URL из текста"""
        
        # Паттерн для поиска URL
        url_pattern = r'https?://[^\s\)]+|www\.[^\s\)]+'
        urls = re.findall(url_pattern, text)
        
        # Очистка URL от лишних символов
        cleaned_urls = []
        for url in urls:
            url = url.rstrip('.,;:!?')
            if url not in cleaned_urls:
                cleaned_urls.append(url)
        
        return cleaned_urls