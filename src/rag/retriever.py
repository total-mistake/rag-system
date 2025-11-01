from src.indexing.indexer import DocumentIndexer
from src.config import settings
from ..models.document import Document
from ..models.search import SearchResult
from ..models.pipeline import VectorSearchResult, StageMetrics
from src.llm.llm_factory import LLMClientsFactory
from ..config.prompts import create_hyde_response
from typing import List
import logging
import time

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self):
        logger.debug("Инициализация индексатора")
        self.indexer = DocumentIndexer(
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            chroma_db_path=settings.chroma_db_path,
            collection_name=settings.collection_name
        )
        logger.debug("Завершение инициализации индексатора")

    def search(self, query: str, top_k: int = 1) -> VectorSearchResult:
        start_time = time.time()
        logger.debug("Начало векторного поиска")
        logger.info(f"Запрос пользователя:\n — {query}")

        if settings.enable_hyde:
            query = self._hyde(query)

        vector_results = self._vector_search(query, settings.initial_candidates)
        vector_search_time = time.time() - start_time

        vector_results.sort(key=lambda x: x.final_score, reverse=True)
        final_results = vector_results[:top_k]

        result = VectorSearchResult(
            StageMetrics(
                stage_name="retriever", 
                duration=vector_search_time),
            final_results
        )
        
        total_time = time.time() - start_time
        # logging.info(f"Поиск завершен: {len(final_results)} результатов за {total_time:.3f}с")
        logger.info(f"Успешный векторный поиск. Время выполнения: {total_time:3f}")
        logger.info(f"Результаты векторного поиска:\n{"\n".join(f"{result.document.url}: {result.vector_score:.3f}" for result in final_results)}")
        
        return result

    def _vector_search(self, query: str, top_k: int = 1) -> List[SearchResult]:
        try:
            results = self.indexer.db_connector.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "embeddings", "metadatas", "distances"]
            )

            search_results = []
            logger.debug(f"Найдено {len(results['ids'][0])} результатов поиска")

            for i, (doc_id, distance, metadata) in enumerate(zip(
                results['ids'][0], 
                results['distances'][0],
                results['metadatas'][0]
            )):
                doc = Document(
                    id=int(doc_id),
                    title=metadata.get('title', ''),
                    url=metadata.get('url', ''),
                    text=metadata.get('text', '')
                )

                search_result = SearchResult(
                    document=doc,
                    vector_score=distance,
                )

                search_results.append(search_result)

            logger.debug(f"Обработано {len(search_results)} результатов поиска")
            return search_results
        except Exception as e:
            logger.error(f"Ошибка при векторном поиске: {e}")
            return []
        
    def _hyde(self, query: str) -> str:
        client = LLMClientsFactory.create_llm_client(settings.llm_client)

        try:
            response = client.chat(
                model = settings.generation_model,
                messages=create_hyde_response(query),
                temperature=0.7
            )

            logger.info(f"Сгенерированный гипотетический документ:\n {response.content}")

            return response.content
        
        except Exception as e:
            logger.error(f"Ошибка при генерации документа: {e}")
            raise