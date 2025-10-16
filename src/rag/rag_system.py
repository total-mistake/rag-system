from src.rag.retriever import DocumentRetriever
from src.rag.reranker import Reranker
from src.rag.generator import ResponseGenerator
from src.models.pipeline import RAGPipeline
from src.config.settings import settings
import time

class RAGSystem:
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.generator = ResponseGenerator()
        self.reranker = None
        self.pipeline = None

        if settings.enable_reranking:
            self.reranker = Reranker()

    def request(self, query: str) -> str:
        self.pipeline = RAGPipeline(query)

        start_time = time.time()

        retriever_results = self.retriever.search(query, top_k=settings.initial_candidates)
        self.pipeline.vector_search = retriever_results

        if self.reranker and len(retriever_results.results) > 1:
            rerank_results = self.reranker.rerank(query, retriever_results.results)
            self.pipeline.reranking = retriever_results
            documents = [result.document for result in rerank_results.results]
        else:
            documents = [result.document for result in retriever_results.results]
        
        response = self.generator.generate_answer(query, documents)
        self.pipeline.generation = response
        self.pipeline.total_duration = time.time() - start_time

        return response.answer
    
    def get_debug_info(self) -> RAGPipeline:
        if self.pipeline:
            return self.pipeline
        else:
            raise Exception("Object RAGPipeline does not exist, please run RAGSystem.request() first.")


        