import chromadb
from ..models.document import DocumentCollection
from typing import List
from chromadb.errors import NotFoundError
import logging

logger = logging.getLogger(__name__)

class ChromaDBManager:

    def __init__(self, db_path, collection_name, model, embed_func):
        self.client = chromadb.PersistentClient(path=db_path) 

        self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embed_func
            )
    
    def add_documents(self, documents: DocumentCollection):
        try:
            logger.info(f"Добавляем {len(documents.documents)} документов в ChromaDB")
            
            self.collection.add(
                ids=[str(d.id) for d in documents.documents],
                documents=[f"{d.title}\n{d.text}" for d in documents.documents],
                metadatas=documents.to_list_of_dict()
            )
            
            logger.info("Документы успешно добавлены в ChromaDB")
            
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов в ChromaDB: {e}")
            raise

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(name=self.collection.name)
            return True
        except NotFoundError as e:
            return False

    def delete_collection(self):
        self.client.delete_collection(name=self.collection.name)

    def get_collection_info(self) -> dict:
        """Возвращает информацию о коллекции"""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }
    
