from src.rag.retriever import DocumentRetriever
from src.reranker.reranker_factory import RerankerProviderFactory
from src.rag.generator import ResponseGenerator
from src.models.pipeline import RAGPipeline
from src.config.settings import settings
import logging
import time

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.generator = ResponseGenerator()
        self.reranker = None
        self.pipeline = None

        if settings.enable_reranking:
            self.reranker = RerankerProviderFactory.create_provider(settings.reranker_provider)

    def request(self, query: str) -> str:
        start_time = time.time()
        logger.debug("Начало обработки запроса")

        self.pipeline = RAGPipeline(query)

        retriever_results = self.retriever.search(query, top_k=settings.initial_candidates)
        self.pipeline.retriever = retriever_results

        if self.reranker and len(retriever_results.results) > 1:
            rerank_results = self.reranker.rerank(query, retriever_results.results)
            self.pipeline.reranking = rerank_results

            filtered_docs = [doc for doc in rerank_results.results if doc.final_score and doc.final_score > settings.document_filtering_threshold]
            documents = [result.document for result in filtered_docs]
        else:
            documents = [result.document for result in retriever_results.results]

        response = self.generator.generate_answer(query, documents)
        self.pipeline.generation = response
        self.pipeline.update_general_results()

        total_time = time.time() - start_time
        self.pipeline.general.total_duration = total_time
        logger.info(f"Успешная обработка запроса. Время выполнения: {total_time:3f}")
        return response.answer
    
    def get_debug_info(self) -> RAGPipeline:
        if self.pipeline:
            return self.pipeline
        else:
            raise Exception("Object RAGPipeline does not exist, please run RAGSystem.request() first.")
        


        