from src.indexing.indexer import DocumentIndexer
from src.indexing.config import IndexingConfig
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Создаем индексатор (коллекция уже проиндексирована)
indexer = DocumentIndexer(
    embedding_provider=IndexingConfig.embedding_provider,
    embedding_model=IndexingConfig.embedding_model,
    chroma_db_path=IndexingConfig.chroma_db_path,
    collection_name=IndexingConfig.collection_name
)

# Проверяем информацию о коллекции
collection_info = indexer.db_connector.get_collection_info()
print(f"Коллекция: {collection_info['name']}")
print(f"Количество документов: {collection_info['count']}")

# Выполняем тестовый поиск
print("\n=== Тестовый поиск ===")
query = "Как подключить онлайн-чат с ассистентом"
print(f"Запрос: {query}")

results = indexer.db_connector.collection.query(
    query_texts=[query],
)

print(f"\nНайдено результатов: {len(results['ids'][0])}")
for i, (doc_id, document, distance) in enumerate(zip(
    results['ids'][0], 
    results['documents'][0], 
    results['distances'][0]
)):
    print(f"\n{i+1}. ID: {doc_id}, Расстояние: {distance:.4f}")
    print(f"Документ: {document[:200]}...")