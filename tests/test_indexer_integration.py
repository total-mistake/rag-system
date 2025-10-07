"""
Интеграционные тесты для индексатора с реальной ChromaDB и векторным поиском
"""
import pytest
import os
import shutil
from pathlib import Path

from src.indexing.indexer import DocumentIndexer
from src.indexing.config import EmbeddingProviderType
from src.models.document import DocumentCollection


class TestDocumentIndexerIntegration:
    """Интеграционные тесты DocumentIndexer с реальной БД"""
    
    def test_indexer_initialization(self, test_chroma_db_path, test_collection_name, test_embedding_model):
        """Тест инициализации индексатора с тестовыми параметрами"""
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name=test_collection_name
        )
        
        assert indexer is not None
        assert indexer.db_connector is not None
        
        # Проверяем, что коллекция создалась
        collection_info = indexer.db_connector.get_collection_info()
        assert collection_info["name"] == test_collection_name
        assert collection_info["count"] == 0  # Пустая коллекция
    
    def test_index_documents_success(self, test_json_file, test_chroma_db_path, test_collection_name, test_embedding_model):
        """Тест успешной индексации документов"""
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name=test_collection_name
        )
        
        # Индексируем документы
        indexer.index_documents(test_json_file)
        
        # Проверяем, что документы добавились
        collection_info = indexer.db_connector.get_collection_info()
        assert collection_info["count"] == 20  # 20 тестовых документов
    
    def test_index_documents_already_exists_no_force(self, test_json_file, test_chroma_db_path, test_collection_name, test_embedding_model):
        """Тест поведения при существующей коллекции без force_reindex"""
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name=test_collection_name
        )
        
        # Первая индексация
        indexer.index_documents(test_json_file)
        initial_count = indexer.db_connector.get_collection_info()["count"]
        
        # Вторая индексация без force_reindex - должна пропуститься
        indexer.index_documents(test_json_file, force_reindex=False)
        
        # Количество документов не должно измениться
        final_count = indexer.db_connector.get_collection_info()["count"]
        assert final_count == initial_count
    
    def test_index_documents_force_reindex(self, test_json_file, test_chroma_db_path, test_collection_name, test_embedding_model):
        """Тест принудительной переиндексации"""
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name=test_collection_name
        )
        
        # Первая индексация
        indexer.index_documents(test_json_file)
        assert indexer.db_connector.get_collection_info()["count"] == 20
        
        # Принудительная переиндексация
        indexer.index_documents(test_json_file, force_reindex=True)
        
        # Документы должны быть переиндексированы
        assert indexer.db_connector.get_collection_info()["count"] == 20
    
    def test_clear_index(self, test_json_file, test_chroma_db_path, test_collection_name, test_embedding_model):
        """Тест очистки индекса"""
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name=test_collection_name
        )
        
        # Индексируем документы
        indexer.index_documents(test_json_file)
        assert indexer.db_connector.get_collection_info()["count"] == 20
        
        # Очищаем индекс
        indexer.clear_index()
        
        # Создаем новый индексатор (так как коллекция была удалена)
        new_indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name=test_collection_name
        )
        
        # Коллекция должна быть пустой
        assert new_indexer.db_connector.get_collection_info()["count"] == 0


class TestVectorSearchFunctionality:
    """Тесты векторного поиска"""
    
    @pytest.fixture(autouse=True)
    def setup_indexed_collection(self, test_json_file, test_chroma_db_path, test_collection_name, test_embedding_model):
        """Автоматически создает проиндексированную коллекцию для каждого теста"""
        self.indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name=test_collection_name
        )
        
        # Индексируем документы если еще не проиндексированы
        if self.indexer.db_connector.get_collection_info()["count"] == 0:
            self.indexer.index_documents(test_json_file)
    
    def test_vector_search_basic(self):
        """Базовый тест векторного поиска"""
        # Поиск по форме заказа
        results = self.indexer.db_connector.collection.query(
            query_texts=["настройка формы заказа товаров"],
            n_results=3
        )
        
        assert len(results["ids"]) > 0
        assert len(results["documents"]) > 0
        assert len(results["metadatas"]) > 0
        assert len(results["distances"]) > 0
        
        # Проверяем, что результаты отсортированы по релевантности (расстоянию)
        distances = results["distances"][0]
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
    
    @pytest.mark.parametrize("test_case", [
        {
            "query": "как настроить форму заказа товаров",
            "expected_doc_id": 0,
            "description": "Поиск по настройке формы заказа"
        },
        {
            "query": "добавить WhatsApp Telegram в шапку",
            "expected_doc_id": 1,
            "description": "Поиск по мессенджерам в шапке"
        },
        {
            "query": "отображение товаров в каталоге на главной",
            "expected_doc_id": 2,
            "description": "Поиск по отображению товаров"
        }
    ])
    def test_vector_search_accuracy(self, test_case):
        """Тест точности векторного поиска - топ-1 результат должен быть ожидаемым документом"""
        query = test_case["query"]
        expected_doc_id = test_case["expected_doc_id"]
        description = test_case["description"]
        
        # Выполняем поиск
        results = self.indexer.db_connector.collection.query(
            query_texts=[query],
            n_results=5  # Берем топ-5 для анализа
        )
        
        # Проверяем, что есть результаты
        assert len(results["ids"]) > 0, f"Нет результатов для запроса: {query}"
        
        # Получаем ID топ-1 результата
        top_result_id = int(results["ids"][0][0])
        
        # Проверяем, что топ-1 результат соответствует ожидаемому
        assert top_result_id == expected_doc_id, (
            f"Тест '{description}' провален.\n"
            f"Запрос: '{query}'\n"
            f"Ожидаемый документ ID: {expected_doc_id}\n"
            f"Получен документ ID: {top_result_id}\n"
            f"Топ-5 результатов: {results['ids'][0][:5]}\n"
            f"Расстояния: {results['distances'][0][:5]}"
        )
    
    def test_vector_search_multiple_queries(self, search_test_cases):
        """Тест векторного поиска с множественными запросами"""
        # Берем первые 5 тестовых случаев для ускорения
        test_cases = search_test_cases[:5]
        
        for test_case in test_cases:
            query = test_case["query"]
            expected_doc_id = test_case["expected_doc_id"]
            description = test_case["description"]
            
            # Выполняем поиск
            results = self.indexer.db_connector.collection.query(
                query_texts=[query],
                n_results=3
            )
            
            # Проверяем, что есть результаты
            assert len(results["ids"]) > 0, f"Нет результатов для: {description}"
            
            # Получаем топ-3 результата
            top_3_ids = [int(id_) for id_ in results["ids"][0][:3]]
            
            # Ожидаемый документ должен быть в топ-3
            assert expected_doc_id in top_3_ids, (
                f"Тест '{description}' провален.\n"
                f"Запрос: '{query}'\n"
                f"Ожидаемый ID {expected_doc_id} не в топ-3: {top_3_ids}"
            )
    
    def test_vector_search_similarity_scores(self):
        """Тест проверки корректности скоров похожести"""
        # Очень специфичный запрос
        specific_query = "форма заказа товаров настройки магазин"
        
        # Общий запрос
        general_query = "сайт интернет"
        
        # Выполняем поиски
        specific_results = self.indexer.db_connector.collection.query(
            query_texts=[specific_query],
            n_results=1
        )
        
        general_results = self.indexer.db_connector.collection.query(
            query_texts=[general_query],
            n_results=1
        )
        
        # Специфичный запрос должен иметь меньшее расстояние (больше похожести)
        specific_distance = specific_results["distances"][0][0]
        general_distance = general_results["distances"][0][0]
        
        # Расстояния должны быть в разумных пределах (0-2 для косинусного расстояния)
        assert 0 <= specific_distance <= 2, f"Некорректное расстояние: {specific_distance}"
        assert 0 <= general_distance <= 2, f"Некорректное расстояние: {general_distance}"
    
    def test_vector_search_empty_query(self):
        """Тест поиска с пустым запросом"""
        with pytest.raises(Exception):
            self.indexer.db_connector.collection.query(
                query_texts=[""],
                n_results=1
            )
    
    def test_vector_search_large_n_results(self):
        """Тест поиска с большим количеством результатов"""
        results = self.indexer.db_connector.collection.query(
            query_texts=["сайт"],
            n_results=25  # Больше чем документов в коллекции
        )
        
        # Должно вернуть максимум доступных документов (20)
        assert len(results["ids"][0]) <= 20
        assert len(results["documents"][0]) <= 20


class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    def test_index_nonexistent_file(self, test_chroma_db_path, test_collection_name, test_embedding_model):
        """Тест индексации несуществующего файла"""
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name=test_collection_name
        )
        
        with pytest.raises(FileNotFoundError):
            indexer.index_documents("nonexistent_file.json")
    
    def test_invalid_json_file(self, test_chroma_db_path, test_collection_name, test_embedding_model, tmp_path):
        """Тест индексации файла с невалидным JSON"""
        # Создаем файл с невалидным JSON
        invalid_json_file = tmp_path / "invalid.json"
        invalid_json_file.write_text("{ invalid json content", encoding='utf-8')
        
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name=test_collection_name
        )
        
        with pytest.raises(Exception):  # JSON decode error
            indexer.index_documents(str(invalid_json_file))


@pytest.mark.ollama
class TestOllamaIntegration:
    """Тесты интеграции с Ollama API"""
    
    def test_ollama_server_availability(self, ollama_base_url, check_ollama_available):
        """Тест доступности Ollama сервера"""
        if not check_ollama_available:
            pytest.skip("Ollama сервер недоступен")
        
        import requests
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
    
    def test_indexer_ollama_initialization(self, request, test_ollama_model, ollama_base_url,
                                         test_chroma_db_path, check_ollama_available,
                                         ollama_models_available):
        """Тест инициализации индексатора с Ollama провайдером"""
        if not request.config.getoption("--run-ollama", default=False):
            pytest.skip("Требует запущенный Ollama сервер с моделью")
        
        if not check_ollama_available:
            pytest.skip("Ollama сервер недоступен")
        
        if test_ollama_model not in ollama_models_available:
            pytest.skip(f"Модель {test_ollama_model} недоступна в Ollama")
        
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.OLLAMA,
            embedding_model=test_ollama_model,
            chroma_db_path=test_chroma_db_path,
            collection_name="test_ollama_collection",
            base_url=ollama_base_url
        )
        
        assert indexer is not None
        assert indexer.db_connector is not None
        
        collection_info = indexer.db_connector.get_collection_info()
        assert collection_info["name"] == "test_ollama_collection"
        assert collection_info["count"] == 0
    
    def test_ollama_document_indexing(self, request, test_json_file, test_ollama_model,
                                    ollama_base_url, test_chroma_db_path,
                                    check_ollama_available, ollama_models_available):
        """Тест индексации документов через Ollama"""
        if not request.config.getoption("--run-ollama", default=False):
            pytest.skip("Требует запущенный Ollama сервер с моделью")
        
        if not check_ollama_available:
            pytest.skip("Ollama сервер недоступен")
        
        if test_ollama_model not in ollama_models_available:
            pytest.skip(f"Модель {test_ollama_model} недоступна в Ollama")
        
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.OLLAMA,
            embedding_model=test_ollama_model,
            chroma_db_path=test_chroma_db_path,
            collection_name="test_ollama_indexing",
            base_url=ollama_base_url
        )
        
        # Индексируем документы
        indexer.index_documents(test_json_file)
        
        # Проверяем, что документы добавились
        collection_info = indexer.db_connector.get_collection_info()
        assert collection_info["count"] == 20
    
    def test_ollama_vector_search(self, request, test_json_file, test_ollama_model, ollama_base_url,
                                test_chroma_db_path, check_ollama_available, ollama_models_available):
        """Тест векторного поиска с Ollama эмбеддингами"""
        if not request.config.getoption("--run-ollama", default=False):
            pytest.skip("Требует запущенный Ollama сервер с моделью")
        
        if not check_ollama_available:
            pytest.skip("Ollama сервер недоступен")
        
        if test_ollama_model not in ollama_models_available:
            pytest.skip(f"Модель {test_ollama_model} недоступна в Ollama")
        
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.OLLAMA,
            embedding_model=test_ollama_model,
            chroma_db_path=test_chroma_db_path,
            collection_name="test_ollama_search",
            base_url=ollama_base_url
        )
        
        # Индексируем документы
        indexer.index_documents(test_json_file)
        
        # Выполняем поиск
        results = indexer.db_connector.collection.query(
            query_texts=["настройка формы заказа товаров"],
            n_results=3
        )
        
        assert len(results["ids"]) > 0
        assert len(results["documents"]) > 0
        assert len(results["metadatas"]) > 0
        assert len(results["distances"]) > 0
        
        # Проверяем, что результаты отсортированы по релевантности
        distances = results["distances"][0]
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
    
    def test_ollama_force_reindex(self, request, test_json_file, test_ollama_model, ollama_base_url,
                                test_chroma_db_path, check_ollama_available, ollama_models_available):
        """Тест принудительной переиндексации с Ollama"""
        if not request.config.getoption("--run-ollama", default=False):
            pytest.skip("Требует флаг --run-ollama")
        if not check_ollama_available:
            pytest.skip("Ollama сервер недоступен")
        
        if test_ollama_model not in ollama_models_available:
            pytest.skip(f"Модель {test_ollama_model} недоступна в Ollama")
        
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.OLLAMA,
            embedding_model=test_ollama_model,
            chroma_db_path=test_chroma_db_path,
            collection_name="test_ollama_reindex",
            base_url=ollama_base_url
        )
        
        # Первая индексация
        indexer.index_documents(test_json_file)
        assert indexer.db_connector.get_collection_info()["count"] == 20
        
        # Принудительная переиндексация
        indexer.index_documents(test_json_file, force_reindex=True)
        
        # Документы должны быть переиндексированы
        assert indexer.db_connector.get_collection_info()["count"] == 20
    
    def test_ollama_connection_error(self, test_ollama_model, test_chroma_db_path):
        """Тест обработки ошибки подключения к Ollama"""
        
        # Используем неверный URL для имитации ошибки подключения
        wrong_url = "http://localhost:99999"
        
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.OLLAMA,
            embedding_model=test_ollama_model,
            chroma_db_path=test_chroma_db_path,
            collection_name="test_error_handling",
            base_url=wrong_url
        )
        
        # Создаем минимальный JSON файл для теста
        import tempfile
        import json
        from src.models.document import Document, DocumentCollection
        
        test_doc = Document(
            id=1,
            title="Test",
            url="http://test.com",
            text="Test text",
            filename="test.md"
        )
        test_collection = DocumentCollection(documents=[test_doc])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_collection.to_json_dict(), f)
            temp_file = f.name
        
        # Должна возникнуть ошибка при индексации из-за недоступности сервера
        with pytest.raises(Exception):  # requests.exceptions.ConnectionError или подобная
            indexer.index_documents(temp_file)
        
        # Очищаем временный файл
        import os
        os.unlink(temp_file)
    
    def test_ollama_search_accuracy(self, request, test_json_file, test_ollama_model, ollama_base_url,
                                  test_chroma_db_path, check_ollama_available, ollama_models_available):
        """Тест точности поиска с Ollama эмбеддингами"""
        if not request.config.getoption("--run-ollama", default=False):
            pytest.skip("Требует флаг --run-ollama")
        if not check_ollama_available:
            pytest.skip("Ollama сервер недоступен")
        
        if test_ollama_model not in ollama_models_available:
            pytest.skip(f"Модель {test_ollama_model} недоступна в Ollama")
        
        indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.OLLAMA,
            embedding_model=test_ollama_model,
            chroma_db_path=test_chroma_db_path,
            collection_name="test_ollama_accuracy",
            base_url=ollama_base_url
        )
        
        # Индексируем документы
        indexer.index_documents(test_json_file)
        
        # Тестируем несколько поисковых запросов
        test_queries = [
            "форма заказа товаров",
            "мессенджеры в шапке сайта",
            "каталог товаров на главной"
        ]
        
        for query in test_queries:
            results = indexer.db_connector.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            # Проверяем, что есть результаты
            assert len(results["ids"]) > 0, f"Нет результатов для запроса: {query}"
            
            # Проверяем, что расстояния в разумных пределах
            distances = results["distances"][0]
            assert all(0 <= d <= 2 for d in distances), f"Некорректные расстояния для запроса: {query}"
            
            # Проверяем сортировку по релевантности
            assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1)), \
                f"Результаты не отсортированы по релевантности для запроса: {query}"


class TestProviderComparison:
    """Тесты сравнения разных провайдеров эмбеддингов"""
    
    def test_local_vs_ollama_initialization(self, test_chroma_db_path, test_embedding_model, test_ollama_model, ollama_base_url):
        """Тест консистентности инициализации между локальным и Ollama провайдером"""
        
        # Создаем локальный индексатор
        local_indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.LOCAL,
            embedding_model=test_embedding_model,
            chroma_db_path=test_chroma_db_path,
            collection_name="local_test"
        )
        
        # Создаем Ollama индексатор (без реальных вызовов API)
        ollama_indexer = DocumentIndexer(
            embedding_provider=EmbeddingProviderType.OLLAMA,
            embedding_model=test_ollama_model,
            chroma_db_path=test_chroma_db_path,
            collection_name="ollama_test",
            base_url=ollama_base_url
        )
        
        # Оба индексатора должны успешно инициализироваться
        assert local_indexer.db_connector is not None
        assert ollama_indexer.db_connector is not None
        
        # Проверяем, что коллекции созданы
        local_info = local_indexer.db_connector.get_collection_info()
        ollama_info = ollama_indexer.db_connector.get_collection_info()
        
        assert local_info["name"] == "local_test"
        assert ollama_info["name"] == "ollama_test"
    
    def test_provider_factory_creation(self, test_embedding_model, test_ollama_model, ollama_base_url):
        """Тест создания провайдеров через фабрику"""
        from src.indexing.embedding import EmbeddingProviderFactory
        from src.indexing.embedding import LocalEmbeddingProvider, OllamaEmbeddingProvider
        
        # Создаем локальный провайдер
        local_provider = EmbeddingProviderFactory.create_provider(
            EmbeddingProviderType.LOCAL,
            test_embedding_model
        )
        assert isinstance(local_provider, LocalEmbeddingProvider)
        assert local_provider.model_name == test_embedding_model
        
        # Создаем Ollama провайдер
        ollama_provider = EmbeddingProviderFactory.create_provider(
            EmbeddingProviderType.OLLAMA,
            test_ollama_model,
            base_url=ollama_base_url
        )
        assert isinstance(ollama_provider, OllamaEmbeddingProvider)
        assert ollama_provider.model_name == test_ollama_model
        assert ollama_provider.base_url == ollama_base_url
    
    def test_invalid_embedding_provider(self, test_chroma_db_path, test_embedding_model):
        """Тест с неподдерживаемым провайдером эмбеддингов"""
        with pytest.raises(ValueError, match="Неподдерживаемый тип провайдера"):
            DocumentIndexer(
                embedding_provider="invalid_provider",  # Неверный тип
                embedding_model=test_embedding_model,
                chroma_db_path=test_chroma_db_path,
                collection_name="test_invalid"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])