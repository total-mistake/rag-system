from src.retrieval.retriever import DocumentRetriever
from src.generation.generator import ResponseGenerator
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Создаем ретривер
retriever = DocumentRetriever()

# Проверяем информацию о коллекции
collection_info = retriever.indexer.db_connector.get_collection_info()
print(f"Коллекция: {collection_info['name']}")
print(f"Количество документов: {collection_info['count']}")

# Выполняем тестовый поиск
print("\n=== Тестовый поиск с реранкингом ===")
query = "Как подключить онлайн-чат с ассистентом"
print(f"Запрос: {query}")

results = retriever.search(query, top_k=3)
generator = ResponseGenerator()
response = generator.generate_answer(query, [result.document for result in results])

print(f"Ответ: {response.answer}")
print("Источники:")
for url in response.source_urls:
    print(url)