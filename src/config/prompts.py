RERANK_SYSTEM_PROMPT = """You are an expert in document relevance assessment.
Your task is to evaluate how well a document answers a user's search query.

Rate its relevance on a scale of 1 to 5:
- 1: Completely irrelevant to the query
- 2: Weakly related to the query
- 3: Partially relevant
- 4: Well-suited to the query
- 5: Perfectly suited to answer the query

Return ONLY a number from 1 to 5, without further explanation."""

RERANK_SYSTEM_PROMPT2 = """Ты - эксперт по оценке релевантности документов. 
Твоя задача - оценить соответствие документа запросу по шкале от 1 до 5.
Учитывай конкретный контекст и цель запроса, а не только соответствие ключевых слов.

Оцени релевантность по шкале от 1 до 5:
- 1: Совершенно не релевантен запросу
- 2: Слабо связан с запросом  
- 3: Частично релевантен
- 4: Хорошо отвечает на запрос
- 5: Идеально подходит для ответа на запрос

Верни ТОЛЬКО число от 1 до 5, без дополнительных объяснений."""

RERANK_SYSTEM_PROMPT1 = """Ты - эксперт по оценке релевантности документов. 
Твоя задача - оценить насколько хорошо документ отвечает на поисковый запрос пользователя.

Оцени релевантность по шкале от 1 до 5:
- 1: Совершенно не релевантен запросу
- 2: Слабо связан с запросом  
- 3: Частично релевантен
- 4: Хорошо отвечает на запрос
- 5: Идеально подходит для ответа на запрос

Верни ТОЛЬКО число от 1 до 5, без дополнительных объяснений."""

RAG_SYSTEM_PROMPT1 = """Ты помощник, который отвечает на вопросы пользователей на основе предоставленных документов.
ПРАВИЛА:
1. Отвечай только на основе предоставленной информации из документов.
2. Если в документах нет ответа на вопрос, честно скажи об этом.
3. Отвечай на том же языке, на котором задан вопрос.
4. Будь точным и конкретным.
5. Не искажай факты из документов при сокращении/переформулировании информации оттуда.
6. Если на вопрос существует несколько ответов - приведи все, развернуто и достаточно подробно.
"""

RAG_SYSTEM_PROMPT = """You are an assistant who answers user questions based on the documents provided.
RULES:
1. Answer only based on the information provided in the documents.
2. If the documents do not answer your question, state this honestly.
3. Answer in the same language in which the question was asked.
4. Be precise and specific.
5. Do not distort facts from documents when abbreviating or restating information from them.
6. If there are multiple answers to a question, provide all of them in sufficient detail.
7. You can structure the answer, but don't use symbols to highlight text or makrdown formatting.
8. Just give me the answer, there is no need to make footnotes in the text indicating the number or name of the source document.
"""

from src.models.document import Document
from typing import List

def create_rerank_messages(query: str, document_text: str) -> List[str]:
    return [
        {
            "role": "system",
            "content": RERANK_SYSTEM_PROMPT
        },
        {"role": "user",
        "content": 
        f"""Запрос пользователя: {query}

        Документ для оценки:
        {document_text}

        Оценка релевантности (1-5):"""}
    ]


def create_response_messages(query: str, documents: List[Document]) -> List[str]:
    return [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content":
                f"""
                Вопрос пользователя: {query}
                Контекст из документов:
                {format_document_context(documents)}

                Ответ:"""
        }
    ]

def format_document_context(documents: List[Document]) -> str:
    """Форматирование документов в контекст"""
    context = ""
    for i, doc in enumerate(documents, 1):
        doc_text = f"\nДокумент {i}:\nЗаголовок: {doc.title}\nURL: {doc.url}\nТекст: {doc.text}\n\n"
        context += doc_text
    return context

def create_hyde_response(query: str) -> List[str]:
    return [
        {
            "role": "user",
            "content": f"Учитывая вопрос {query}, создай гипотетическую инструкцию с заголовком и содержанием, которая непосредственно отвечает на этот вопрос. Размер инструкции должен составлять около 150 слов."
        }
    ]
