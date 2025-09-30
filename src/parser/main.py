from src.parser.md_parser import MarkdownParser
from pathlib import Path
from ..models.document import DocumentCollection
import logging
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Пути
    data_dir = Path('data')
    storage_dir = Path("storage/documents")
    output_file = storage_dir / "documents.json"

    storage_dir.mkdir(parents=True, exist_ok=True)

    parser = MarkdownParser()

    collection = DocumentCollection()
    doc_id = 0

    logger.info(f"Начинаем парсинг файлов из {data_dir}")

    # Парсим файлы
    for file_path in data_dir.iterdir():
        if not parser.supports_extension(file_path.suffix):
            continue
        try:
            document = parser.parse(file_path, doc_id)
            collection.add_document(document)
            doc_id += 1
        except Exception as e:
            logger.error(f"Ошибка при парсинге файла {file_path.name}: {e}")
            continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(collection.to_json_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"Парсинг завершен. Обработано {len(collection.documents)} документов")
    logger.info(f"Результат сохранен в {output_file}")

if __name__ == "__main__":
    main()