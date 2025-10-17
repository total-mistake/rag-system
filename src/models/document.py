from pydantic import BaseModel, Field, HttpUrl, ConfigDict
from typing import Optional
from datetime import datetime
from typing import List
import json

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

    model_config = ConfigDict(
        str_to_lower=False,
        validate_assignment=True
    )

    id: int = Field(..., ge=0, description="Индекс в FAISS (начинается с 0)")
    title: str = Field(..., min_length=1, description="Заголовок документа")
    url: str = Field(..., description="URL страницы документа")
    text: str = Field(..., min_length=1, description="Основной текст документа")
    filename: Optional[str] = Field(None, min_length=1, description="Имя исходного MD файла")
    created_at: Optional[datetime] = Field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        d = {
            'id': self.id, 
            'title': self.title, 
            'url': str(self.url), 
            'text':self.text, 
            'filename': self.filename,
            'created_at': self.created_at.isoformat()
        }

        return d

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
    
    def to_list_of_dict(self) -> List[dict]:
        return [doc.to_dict() for doc in self.documents]

    @classmethod
    def from_json_file(cls, file_path: str) -> 'DocumentCollection':
        """Загружает коллекцию документов из JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = [Document(**doc_data) for doc_data in data['documents']]
        return cls(documents=documents)
