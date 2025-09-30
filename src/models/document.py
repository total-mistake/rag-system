from pydantic import BaseModel, Field, HttpUrl
from typing import Optional
from datetime import datetime

class Document(BaseModel):
    """
    Представление одного документа в системе.
    
    Attributes:
        id: Уникальный индекс документа (позиция в FAISS)
        title: Заголовок документа
        url: URL страницы документа
        text: Основной текст документа
        filename: Имя исходного файла (для отладки)
        created_at: Время индексации
    """

    id: int = Field(..., ge=0, description="Индекс в FAISS (начинается с 0)")
    title: str = Field(..., min_length=1, description="Заголовок документа")
    url: HttpUrl = Field(..., description="URL страницы документа")
    text: str = Field(..., min_length=1, description="Основной текст документа")
    filename: str = Field(..., min_length=1, description="Имя исходного MD файла")
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        # Разрешаем преобразование типов (например, str → HttpUrl)
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            HttpUrl: lambda v: str(v)
        }

class DocumentCollection(BaseModel):
    """
    Коллекция всех документов с метаданными.
    
    Это то, что сохраняется в documents.json

    Attributes:
        documents: Список документов
    """

    documents: list[Document] = Field(default_factory=list)

    def add_document(self, doc: Document) -> None:
        """Добавляет документ в коллекцию"""
        self.documents.append(doc)

    def get_by_id(self, id: int) -> Optional[Document]:
        """Получает документ по его индексу"""
        for doc in self.documents:
            if doc.id == id:
                return doc
        return None
    
    def to_json_dict(self) -> dict:
        """Конвертирует в словарь для JSON"""
        return {
            "documents": [doc.model_dump(mode='json') for doc in self.documents]
        }