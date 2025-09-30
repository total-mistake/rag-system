"""
Парсер для Markdown файлов с инструкциями.

Формат файла:
    # Заголовок
    **URL:** https://example.com/page
    Текст документа...
"""

import re
from pathlib import Path
from ..models.document import Document
import logging

logger = logging.getLogger(__name__)


class MarkdownParser:
    """
    Парсер Markdown файлов с инструкциями.
    
    Attributes:
        md_file: Путь к файлу Markdown
    """

    def parse(self, file_path: str | Path, doc_id: int) -> Document:
        """
        Парсит MD файл и возвращает объект Document.
        
        Args:
            file_path: Путь к MD файлу
            doc_id: ID документа для FAISS индекса
            
        Returns:
            Document объект
        """

        logger.debug(f"Парсинг файла: {file_path}")
        file_path = Path(file_path)

        try:
            content = file_path.read_text(encoding="utf-8")

            # Извлекаем компоненты
            title = self._extract_title(content)
            url = self._extract_url(content)
            text = self._extract_text(content)

            return Document(
                id=doc_id,
                title=title,
                url=url,
                text=text,
                filename=file_path.name
            )
        except Exception as e:
            logger.error(f"Ошибка при парсинге файла {file_path}: {e}")
            raise
    
    def _extract_title(self, content: str) -> str:
        """
        Извлекает заголовок (первый H1).
        
        Ищет строку вида: # Заголовок
        """
        # Ищем строку, начинающуюся с # (но не ##)
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        
        if not match:
            raise ValueError("Заголовок (H1) не найден в документе")
        
        title = match.group(1).strip()
        return title
    
    def _extract_url(self, content: str) -> str:
        """
        Извлекает URL из строки **URL:** https://...
        
        Примеры валидных форматов:
        - **URL:** https://example.com
        - **URL**: https://example.com
        - **url:** https://example.com
        """
        pattern = r'\*\*URL[:\s]+\*\*\s*(https?://[^\s\n]+)'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if not match:
            raise ValueError("URL не найден в документе (ожидается формат: **URL:** https://...)")
        
        url = match.group(1).strip()
        return url
    
    def _extract_text(self, content: str) -> str:
        """
        Извлекает основной текст документа.
        
        Удаляет:
        - Заголовок (первый H1)
        - Строку с URL
        - Лишние пустые строки
        """
        # Удаляем заголовок
        content = re.sub(r'^#\s+.+$', '', content, count=1, flags=re.MULTILINE)
        
        # Удаляем строку с URL
        content = re.sub(r'\*\*URL[:\s]+\*\*\s*https?://[^\s\n]+', '', content, flags=re.IGNORECASE)
        
        # Убираем лишние пустые строки (оставляем максимум одну)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Убираем пробелы в начале и конце
        text = content.strip()
        
        if not text:
            raise ValueError("Текст документа пуст после парсинга")
        
        return text
    
    def supports_extension(self, extension: str) -> bool:
        """Проверяет, поддерживается ли расширение файла"""
        return extension.lower() in ['.md', '.markdown']
    
