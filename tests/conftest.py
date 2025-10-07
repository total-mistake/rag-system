"""
Конфигурация pytest для интеграционных тестов индексатора
"""
import pytest
import tempfile
import os
import json
import shutil
import requests
from datetime import datetime
from pathlib import Path

from src.models.document import Document, DocumentCollection
from src.indexing.config import EmbeddingProviderType


@pytest.fixture(scope="session")
def test_documents_small():
    """Фикстура с небольшим набором тестовых документов (20 штук)"""
    documents = [
        Document(
            id=0,
            title="Как изменить форму заказа товаров для клиентов?",
            url="https://nethouse.ru/about/instructions/kak_izmenit_formu_zakaza",
            text="Перейдите в раздел Магазин - Настройки - блок Заказы. Вы можете изменить названия полей, сделать их обязательными или необязательными, поменять местами, добавить новые поля.",
            filename="kak_izmenit_formu_zakaza.md"
        ),
        Document(
            id=1,
            title="Как добавить иконки мессенджеров в шапку сайта?",
            url="https://nethouse.ru/about/instructions/kak_dobavit_ikonki_messendzherov_v_shapku_sajta",
            text="Авторизуйтесь в панели управления сайта и перейдите в редактор. Добавьте блок Контакты. Добавьте номер телефона для мессенджера WhatsApp, Telegram, Viber и поставьте галочку напротив пункта Показать в шапке сайта.",
            filename="kak_dobavit_ikonki_messendzherov_v_shapku_sajta.md"
        ),
        Document(
            id=2,
            title="Как показывать в корне каталога товары с главной?",
            url="https://nethouse.ru/about/instructions/kak_pokazyvat_v_korne_kataloga_tovari_s_glavnoy",
            text="Перейдите в раздел Магазин - Настройки - блок Внешний вид каталога товаров на сайте. Включите отображение товаров. Установите значение в поле вывода количества товаров на страницу.",
            filename="kak_pokazyvat_v_korne_kataloga_tovari_s_glavnoy.md"
        ),
        Document(
            id=3,
            title="Нужен ли мне сайт?",
            url="https://nethouse.ru/about/instructions/nuzhen_li_mne_sajt",
            text="Создание представительства в интернете открывает для вас большие возможности для развития бизнеса и привлечения новых клиентов. Интернет-магазин, сайт-визитка, корпоративный сайт, сайт специалиста, блог, лендинг, портфолио.",
            filename="nuzhen_li_mne_sajt.md"
        ),
        Document(
            id=4,
            title="Как добавить блок Отзывы и комментарии на главную страницу?",
            url="https://nethouse.ru/about/instructions/dobavit_blok_otzyvy_na_glavnyu",
            text="Перейдите в раздел Отзывы и комментарии, нажмите Редактировать. Установите галочку на Показывать на главной странице. Нажмите Сохранить.",
            filename="dobavit_blok_otzyvy_na_glavnyu.md"
        ),
        Document(
            id=5,
            title="Как получить API Key для Яндекс.Карт?",
            url="https://nethouse.ru/about/instructions/kak_poluchit_apikey_dlya_kart_yandex",
            text="Перейдите в Кабинет разработчика. Нажмите кнопку Подключить API. Выберите сервис JavaScript API и HTTP Геокодер. Заполните анкету — ваш API-ключ будет сразу готов к использованию.",
            filename="kak_poluchit_apikey_dlya_kart_yandex.md"
        ),
        Document(
            id=6,
            title="Как изменить адрес страницы услуги?",
            url="https://nethouse.ru/about/instructions/kak_izmenit_adres_stranicy_uslugi",
            text="ЧПУ можно прописать только для карточек товара, разделов товара, страниц услуг, статей и отдельных текстовых страниц. Откройте страницу услуги, нажмите Редактировать. Под строкой названия услуги расположено поле Url услуги.",
            filename="kak_izmenit_adres_stranicy_uslugi.md"
        ),
        Document(
            id=7,
            title="Как сменить тарифный план?",
            url="https://nethouse.ru/about/instructions/kak_smenit_tarifnyj_plan",
            text="Переход с тарифного плана Магазин на тарифный план Сайт возможен по окончании оплаченного периода. Для смены тарифного плана с младшего на старший перейдите в раздел Платные услуги - Тарифы.",
            filename="kak_smenit_tarifnyj_plan.md"
        ),
        Document(
            id=8,
            title="Купил домен у вас, но он не работает",
            url="https://nethouse.ru/about/instructions/kupil_domen_u_vas_no_on_ne_rabotaet",
            text="Процедура регистрации и делегирования домена стандартная, необходимо подождать до 48 часов. Предварительно не забудьте оплатить платный тариф.",
            filename="kupil_domen_u_vas_no_on_ne_rabotaet.md"
        ),
        Document(
            id=9,
            title="Как добавить видеофон на слайдер?",
            url="https://nethouse.ru/about/instructions/kak_dobavit_videofon_na_slajder",
            text="Видеофон для слайдера доступен на шаблонах: Новый, Универсальный №1 и №2. Внизу слайдера нажмите на кнопку Редактировать слайд или иконку карандаша. Загрузите видеофон к слайду, вставьте ссылку на ролик с YouTube.",
            filename="kak_dobavit_videofon_na_slajder.md"
        ),
        Document(
            id=10,
            title="Как добавить описание к разделу Услуги?",
            url="https://nethouse.ru/about/instructions/dobavit_opisanie_k_uslugam",
            text="Зайдите в раздел Услуги. Нажмите кнопку Редактировать. Нажмите Описание раздела. Заполните поле в редакторе. Нажмите Сохранить.",
            filename="dobavit_opisanie_k_uslugam.md"
        ),
        Document(
            id=11,
            title="Как настроить цели в Яндекс.Метрике?",
            url="https://nethouse.ru/about/instructions/kak_nastroit_celi_v_yandeks_metrike",
            text="Совместно с Яндекс.Метрикой мы реализовали автоматическое создание целей для отслеживания основных событий на сайте, например, заказа, покупки или отправки форм обратной связи. Откройте Яндекс.Метрику, нажмите кнопку редактирования счетчика.",
            filename="kak_nastroit_celi_v_yandeks_metrike.md"
        ),
        Document(
            id=12,
            title="Как подключить почту к своему домену?",
            url="https://nethouse.ru/about/instructions/podklyuchit_pochtu_k_domenu",
            text="Вы можете подключить к своему домену корпоративную почту. Просто, удобно и бесплатно. Благодаря совместному решению Nethouse и VK WorkSpace. Подключите платный тариф. Перейдите в раздел Приложения - VK WorkSpace.",
            filename="podklyuchit_pochtu_k_domenu.md"
        ),
        Document(
            id=13,
            title="Как подключить и продлить платный тариф?",
            url="https://nethouse.ru/about/instructions/kak_oplatit_tarif_biznes",
            text="Тарифные планы Сайт, Магазин и Профессионал дают вам еще больше возможностей для развития. Пополните баланс сайта в зависимости от желаемого срока подключения тарифа. Перейдите в раздел Настройки — Платные услуги — Тарифы.",
            filename="kak_oplatit_tarif_biznes.md"
        ),
        Document(
            id=14,
            title="Как установить минимальную сумму заказа?",
            url="https://nethouse.ru/about/instructions/kak_ustanovit_minimalnuyu_summu_zakaza",
            text="Данная опция идеально подходит интернет-магазинам, занимающимся продажей небольших и недорогих вещей: сувениров, ручек, зажигалок, магнитов. Перейдите в раздел Магазин - Настройки - блок Заказы. Установите минимальную сумму заказа.",
            filename="kak_ustanovit_minimalnuyu_summu_zakaza.md"
        ),
        Document(
            id=15,
            title="Как добавить к товару поле для промокода?",
            url="https://nethouse.ru/about/instructions/kak_vklyuchit_promokody",
            text="Промокод — это уникальный набор символов, который клиент вводит при покупке и получает скидку. Подключите платный тариф Профессионал или Магазин. Перейдите в раздел Настройки — Магазин — Промокоды — вкладка Активные.",
            filename="kak_vklyuchit_promokody.md"
        ),
        Document(
            id=16,
            title="Как включить корзину?",
            url="https://nethouse.ru/about/instructions/kak_vklyuchit_korzinu",
            text="Перейдите в раздел Магазин - Настройки - блок Корзина. Установите переключатель в положение Вкл, чтобы он стал зеленого цвета. Также выполните другие настройки.",
            filename="kak_vklyuchit_korzinu.md"
        ),
        Document(
            id=17,
            title="Какой браузер вы рекомендуете?",
            url="https://nethouse.ru/about/instructions/kakoj_brauzer_vy_rekomenduete",
            text="Для редактирования сайта мы рекомендуем следующие браузеры: Google Chrome, Mozilla Firefox.",
            filename="kakoj_brauzer_vy_rekomenduete.md"
        ),
        Document(
            id=18,
            title="Какими способами можно купить баллы?",
            url="https://nethouse.ru/about/instructions/kakimi_sposobami_mozhno_kupit_bally",
            text="В данный момент вы можете купить баллы следующими способами: банковские карты Visa MasterCard Maestro, ЮMoney, безналичный расчет для юридических лиц, оплата в терминалах и кассах Связной, Евросеть, Сбербанк, SberPay, СберБизнес.",
            filename="kakimi_sposobami_mozhno_kupit_bally.md"
        ),
        Document(
            id=19,
            title="Как настроить разделение прав доступа к сайту?",
            url="https://nethouse.ru/about/instructions/kak_nastroit_razdelenie_prav_dostupa_k_sajtu",
            text="Вы можете предоставить своим сотрудникам, Агентам или другим специалистам ограниченный и безопасный доступ к сайту с различным функционалом. Подключите платный тариф Магазин или Профессионал. Перейдите в раздел Настройки - Пользователи.",
            filename="kak_nastroit_razdelenie_prav_dostupa_k_sajtu.md"
        )
    ]
    
    return DocumentCollection(documents=documents)


@pytest.fixture(scope="session")
def test_json_file(test_documents_small, tmp_path_factory):
    """Создает временный JSON файл с тестовыми документами"""
    temp_dir = tmp_path_factory.mktemp("test_data")
    json_path = temp_dir / "test_documents.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_documents_small.to_json_dict(), f, ensure_ascii=False, indent=2)
    
    return str(json_path)


@pytest.fixture(scope="session")
def test_chroma_db_path(tmp_path_factory):
    """Создает временную директорию для тестовой ChromaDB"""
    temp_dir = tmp_path_factory.mktemp("test_chroma_db")
    return str(temp_dir)


@pytest.fixture(scope="session")
def test_collection_name():
    """Имя тестовой коллекции"""
    return "test_documents_collection"


@pytest.fixture(scope="session") 
def test_embedding_model():
    """Модель эмбеддингов для тестов (быстрая и легкая)"""
    return "jinaai/jina-embeddings-v3"


@pytest.fixture(scope="session")
def test_ollama_model():
    """Модель Ollama для тестов"""
    return "qwen3-embedding:0.6b"


@pytest.fixture(scope="session")
def ollama_base_url():
    """URL для Ollama API"""
    return "http://172.16.100.164:11434"


@pytest.fixture(scope="session")
def check_ollama_available(ollama_base_url):
    """Проверяет доступность Ollama сервера"""
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


@pytest.fixture(scope="session")
def ollama_models_available(ollama_base_url, check_ollama_available):
    """Получает список доступных моделей в Ollama"""
    if not check_ollama_available:
        return []
    
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
    except:
        pass
    return []


@pytest.fixture
def indexer_factory():
    """Фабрика для создания индексаторов с разными провайдерами"""
    def _create_indexer(provider_type, model, chroma_db_path, collection_name, **kwargs):
        from src.indexing.indexer import DocumentIndexer
        return DocumentIndexer(
            embedding_provider=provider_type,
            embedding_model=model,
            chroma_db_path=chroma_db_path,
            collection_name=collection_name,
            **kwargs
        )
    return _create_indexer


# Тестовые запросы для проверки векторного поиска
@pytest.fixture
def search_test_cases():
    """Тестовые случаи для проверки векторного поиска"""
    return [
        {
            "query": "как настроить форму заказа товаров",
            "expected_doc_id": 0,  # "Как изменить форму заказа товаров для клиентов?"
            "description": "Поиск по настройке формы заказа"
        },
        {
            "query": "добавить WhatsApp Telegram в шапку",
            "expected_doc_id": 1,  # "Как добавить иконки мессенджеров в шапку сайта?"
            "description": "Поиск по мессенджерам в шапке"
        },
        {
            "query": "отображение товаров в каталоге на главной",
            "expected_doc_id": 2,  # "Как показывать в корне каталога товары с главной?"
            "description": "Поиск по отображению товаров"
        },
        {
            "query": "создание сайта интернет магазина",
            "expected_doc_id": 3,  # "Нужен ли мне сайт?"
            "description": "Поиск по созданию сайта"
        },
        {
            "query": "отзывы комментарии на главной странице",
            "expected_doc_id": 4,  # "Как добавить блок Отзывы и комментарии на главную страницу?"
            "description": "Поиск по отзывам на главной"
        },
        {
            "query": "API ключ для Яндекс карт",
            "expected_doc_id": 5,  # "Как получить API Key для Яндекс.Карт?"
            "description": "Поиск по API ключу Яндекс карт"
        },
        {
            "query": "промокод скидка товары",
            "expected_doc_id": 15,  # "Как добавить к товару поле для промокода?"
            "description": "Поиск по промокодам"
        },
        {
            "query": "корзина товаров включить",
            "expected_doc_id": 16,  # "Как включить корзину?"
            "description": "Поиск по включению корзины"
        },
        {
            "query": "Chrome Firefox браузер рекомендации",
            "expected_doc_id": 17,  # "Какой браузер вы рекомендуете?"
            "description": "Поиск по рекомендациям браузера"
        },
        {
            "query": "купить баллы Visa MasterCard оплата",
            "expected_doc_id": 18,  # "Какими способами можно купить баллы?"
            "description": "Поиск по способам оплаты"
        }
    ]


# Настройки pytest
def pytest_configure(config):
    """Конфигурация pytest"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests with real DB"
    )
    config.addinivalue_line(
        "markers", "search: marks tests that test vector search functionality"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (they use real embedding models)"
    )
    config.addinivalue_line(
        "markers", "ollama: marks tests that require Ollama server"
    )
    config.addinivalue_line(
        "markers", "local_only: marks tests that only work with local models"
    )


def pytest_collection_modifyitems(config, items):
    """Автоматически помечает тесты маркерами"""
    for item in items:
        # Все тесты в этом наборе - интеграционные
        item.add_marker(pytest.mark.integration)
        item.add_marker(pytest.mark.slow)
        
        # Помечаем тесты поиска
        if "search" in item.name.lower() or "vector" in item.name.lower():
            item.add_marker(pytest.mark.search)
            
        # Помечаем Ollama тесты
        if "ollama" in item.name.lower():
            item.add_marker(pytest.mark.ollama)


def pytest_addoption(parser):
    """Добавляем опции командной строки для тестов"""
    parser.addoption(
        "--run-ollama",
        action="store_true",
        default=False,
        help="Запускать тесты, требующие реальный Ollama сервер"
    )