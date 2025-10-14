from typing import Optional, List, Dict
from dataclasses import dataclass
import logging
import requests

@dataclass
class OllamaResponse:
    """Унифицированный ответ от Ollama"""
    content: str                                # Содержимое ответа модели
    model: str                                  # Название использованной модели
    total_duration: Optional[int] = None        # Общее время выполнения запроса (с загрузкой модели)
    load_duration: Optional[int] = None         # Время загрузки модели
    eval_duration: Optional[int] = None         # Время генерации ответа
    prompt_eval_count: Optional[int] = None     # Количество токенов в запрос (входные токены)
    eval_count: Optional[int] = None            # Количество токенов в ответе (выходные токены)

class OllamaClient:
    """Клиент для работы с Ollama"""

    def __init__(self, base_url:str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        top_k: int,
        stream: bool = False,
        **kwargs
    ) -> OllamaResponse:
        """Основной метод для чата с моделью"""
        
        try:
            # Формируем payload для Ollama API
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    **kwargs
                }
            }

            self.logger.debug(f"Отправка запроса к ollama: {payload}")

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()
            self.logger.debug(f"Получен ответ от ollama: {result}")

            if result.get("done"):
                return OllamaResponse(
                    content=result.get("message", {}).get("content", "").strip(),
                    model=result.get("model", model),
                    load_duration=result.get("load_duration"),
                    eval_duration=result.get("eval_duration"),
                    total_duration=result.get("total_duration"),
                    prompt_eval_count=result.get("prompt_eval_count"),
                    eval_count=result.get("eval_count")
                )
            else:
                raise RuntimeError("Ollama response is not done")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка при отправке запроса к ollama: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"Неизвестная ошибка: {e}")
            raise e


    def is_healthy(self) -> bool:
        """Проверка доступности сервера"""
        
        try:
            response = requests.get(self.base_url)
            if 200 <= response.status_code < 300:
                self.logger.info(f"Сервер Ollama по адресу {self.base_url} доступен")
                return True
            else:
                self.logger.error(f"Сервер Ollama по адресу {self.base_url} недоступен")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.exception(f"Произошла ошибка при запросе к {self.base_url}: {e}")
            return False
