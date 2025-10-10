from src.retrieval.retriever import DocumentRetriever

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

print(f"\nНайдено результатов: {len(results)}")
for i, result in enumerate(results):
    print(f"\n{i+1}. ID: {result.document.id}")
    print(f"   Заголовок: {result.document.title}")
    print(f"   Векторный скор: {result.vector_score:.4f}")
    print(f"   Реранк скор: {result.rerank_score}")
    print(f"   Финальный скор: {result.final_score:.4f}")