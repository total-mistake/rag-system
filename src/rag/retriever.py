from src.indexing.indexer import DocumentIndexer
from src.config import settings
from ..models.document import Document
from ..models.search import SearchResult
from ..models.pipeline import VectorSearchResult, StageMetrics
from typing import List
import logging
import time

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self):
        self.indexer = DocumentIndexer(
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            chroma_db_path=settings.chroma_db_path,
            collection_name=settings.collection_name
        )

    def search(self, query: str, top_k: int = 1) -> VectorSearchResult:
        start_time = time.time()

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
        logger.info(f"Поиск завершен: {len(final_results)} результатов за {total_time:.3f}с")
        
        return result

    def _vector_search(self, query: str, top_k: int = 1) -> List[SearchResult]:
        try:
            results = self.indexer.db_connector.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "embeddings", "metadatas", "distances"]
            )

            search_results = []

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

            return search_results
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []