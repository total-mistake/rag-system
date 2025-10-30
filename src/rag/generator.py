from src.llm.llm_factory import LLMClientsFactory
from ..models.pipeline import GenerationResult, StageMetrics
from ..models.document import Document
from ..config.prompts import create_response_messages
from ..config.settings import settings

from typing import List
import logging
import time

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Генератор ответов на основе найденных документов"""

    def __init__(self):
        self.client = LLMClientsFactory.create_llm_client(settings.llm_client)

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
            logger.debug("Начало этапа генерации ответа от LLM")

            selected_docs = documents[:settings.max_context_documents]
            source_urls = [doc.url for doc in selected_docs]

            response = self.client.chat(
                model=self.model,
                messages=create_response_messages(query, selected_docs),
                temperature=self.temperature,
                # top_p=self.top_p,
                # top_k=self.top_k
            )

            logger.debug(f"Сгенерированный ответ:\n {response.content}")

            # Парсинг ответа и извлечение URL
            answer = response.content

            logger.debug(f"Генерация ответа: {response.prompt_eval_count} входных + {response.eval_count} выходных токенов.\nВремя загрузки: {response.load_duration}\nВремя генерации ответа: {response.eval_duration}\nОбщее время: {response.total_duration}")

            total_time = time.time() - start_time
            logger.info(f"Успешная генерация ответа от llm. Время выполнения: {total_time:3f}")
            logger.info(f"Ответ RAG-системы:\n — {answer}")

            return GenerationResult(
                metrics=StageMetrics("generation", total_time),
                answer=answer,
                source_urls=source_urls,
                source_documents=selected_docs,
                input_tokens=response.prompt_eval_count,
                output_tokens=response.eval_count,
                total_tokens=response.prompt_eval_count + response.eval_count,
                load_duration=response.load_duration / 1_000_000_000 if response.load_duration is not None else None,
                eval_duration=response.eval_duration / 1_000_000_000 if response.eval_duration is not None else None,
                model_used=response.model
            )

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            raise