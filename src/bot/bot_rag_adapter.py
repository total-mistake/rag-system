from src.rag.rag_system import RAGSystem
from src.models.pipeline import RAGPipeline  # если у тебя классы вынесены
from dataclasses import asdict, is_dataclass

class RAGAdapter:
    def __init__(self):
        self.rag = RAGSystem()
        self.context: RAGPipeline | None = None

        # соответствие внутренних ключей и отображаемых в меню названий
        self.MODULE_NAMES = {
            "retriever": "Извлечение документов",
            "reranker": "Реранжирование",
            "generator": "Генерация ответа",
            "general": "Общая статистика"
        }

        self.PARAM_NAMES = {
            "retriever": {
                "duration": "Время выполнения",
                "results": "Top-n найденных документов",
            },
            "reranker": {
                "duration": "Время выполнения",
                "results": "Top-n найденных документов",
                "total_tokens_used": "Использовано токенов",
            },
            "generator": {
                "duration": "Время генерации",
                "model_used": "Используемая модель",
                "total_tokens": "Всего токенов",
                "input_tokens": "Входные токены",
                "output_tokens": "Выходные токены",
                "source_documents": "Инструкции откуда была взята информация"
            },
             "general": {
                "total_duration": "Общее время выполнения"
            }
        }

    def answer_question(self, question: str) -> str:
        """Получает ответ от RAG и сохраняет контекст выполнения"""
        response = self.rag.request(question)
        self.context = self.rag.get_debug_info()

        if not self.context or not self.context.generation:
            return response

        sources = self.context.generation.source_urls or []
        sources_str = "\n".join(sources) if sources else "—"
        return f"{response}\n\nИсточники:\n{sources_str}"

    def get_all_debug_info(self) -> dict:
        """Преобразует весь контекст в словарь"""
        if not self.context:
            return {}

        if is_dataclass(self.context):
            return asdict(self.context)
        return self.context

    def get_module_info(self, module: str) -> dict:
        """Возвращает словарь с информацией по конкретному этапу RAG"""
        data = self.get_all_debug_info()

        mapping = {
            "retriever": "vector_search",
            "reranker": "reranking",
            "generator": "generation",
            "general": "total_duration"
        }

        key = mapping.get(module.lower())
        if module.lower() == "general":
            return {"total_duration": data.get("total_duration")}

        key = mapping.get(module.lower())
        if not key:
            return {}

        return data.get(key, {})
        
    def get_param_info(self, module: str, param: str):
        """Возвращает значение конкретного параметра"""
        info = self.get_module_info(module)
        if not info:
            return None
        if param in info:
            return info[param]
        elif "metrics" in info and param in info["metrics"]:
            return info["metrics"][param]
        return None

    def format_param(self, module: str, param: str, value) -> str:
        """Форматирует отдельный параметр в человеко-читаемый вид"""
        name = self.PARAM_NAMES.get(module, {}).get(param, param)

        # время выполнения
        if param in ("duration", "total_duration") and isinstance(value, (float, int)):
            return f"{name}: {value:.3f} сек."

        # список документов
        if param in ("results", "source_documents") and isinstance(value, list):
            if not value:
                return f"{name}: —"

            sorted_items = value.copy()
            if module == "retriever":
            # Для ретривера: сортировка по убыванию векторного скора (лучшие сверху)
                sorted_items.sort(key=lambda x: x.get("vector_score", 0), reverse=True)
            elif module == "reranker":
                # Для реранкера: сортировка по убыванию оценки реранкера (лучшие сверху)
                sorted_items.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            lines = [f"*{name}:*"]
            for i, item in enumerate(value[:5], 1):  # максимум 5 документов для читаемости
                doc = item.get("document", {})
                if hasattr(doc, "title"):
                    title = getattr(doc, "title", "Без названия")
                    url = getattr(doc, "url", None)
                else:
                    title = doc.get("title", "Без названия")
                    url = doc.get("url", "")

                score = (
                    item.get("vector_score") if module == "retriever" 
                    else item.get("rerank_score")
                )
                if score is not None:
                    if module == "retriever":
                        score_str = f" — score: {score:.3f}"  # векторный скор с 3 знаками
                    else:  # reranker
                        score_str = f" — score: {int(score)}"  # оценка реранкера без знаков после запятой
                else:
                    score_str = ""
                    
                if url:
                    lines.append(f"{i}. [{title}]({url}){score_str}")
                else:
                    lines.append(f"{i}. {title}{score_str}")

            return "\n".join(lines)

        # обычные числовые значения
        if isinstance(value, (float, int)):
            return f"{name}: {value}"

        # строка
        if isinstance(value, str):
            return f"{name}: {value}"

        return f"{name}: {value}"

    def format_debug_info(self, module: str = None, param: str = None) -> str:
        """Главная функция форматирования"""
        if not self.context:
            return "Нет данных для отображения. Сначала задай вопрос."

        # Отдельный параметр
        if module and param:
            value = self.get_param_info(module, param)
            if value is None:
                return "Нет данных"
            return self.format_param(module, param, value)

        # Один модуль
        if module:
            info = self.get_module_info(module)
            if not info:
                return f"Нет данных по модулю '{module}'"
            title = self.MODULE_NAMES.get(module, module.upper())
            lines = [f"*{title}*"]
            for param_key in self.PARAM_NAMES[module].keys():
                value = self.get_param_info(module, param_key)
                if value is not None:
                    lines.append(self.format_param(module, param_key, value))
            return "\n".join(lines)

        # Общая сводка
        data = self.get_all_debug_info()
        total = data.get("total_duration", None)
        text = ["*Сводка по RAG-пайплайну:*"]
        text.append(f"Запрос: _{data.get('query', '')}_")
        if total:
            text.append(f"Общее время: {total:.3f} сек.\n")

        mapping = {
            "retriever": "vector_search",
            "reranker": "reranking",
            "generator": "generation",
        }

        for module, key in mapping.items():
            mod = data.get(key)
            if not mod:
                continue
            duration = (
                mod.get("metrics", {}).get("duration")
                if "metrics" in mod else mod.get("duration")
            )
            module_title = self.MODULE_NAMES.get(module, module)
            text.append(f"{module_title}: {duration:.3f} сек.")

        return "\n".join(text)