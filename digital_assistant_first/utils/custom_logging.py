import logging
from datetime import datetime
from pathlib import Path
import json


def setup_logging(logging_path="logs/digital_assistant.log"):
    """Настройка логирования"""
    # Создаем директорию для логов, если она не существует
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Получаем логгер с именем "digital_assistant"
    logger = logging.getLogger("digital_assistant")
    logger.setLevel(logging.INFO)

    # Очищаем существующие обработчики (если они есть), чтобы избежать дублирования
    if logger.hasHandlers():
        logger.handlers.clear()

    # Задаем формат логов
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Создаем обработчик для записи в файл
    file_handler = logging.FileHandler(logging_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Создаем обработчик для вывода в консоль
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def log_api_call(
    logger: logging.Logger, source: str, request: str, response: str, error: str = None
):
    """
    Логирование API вызовов

    Args:
        source: Источник запроса (LLM/SerpAPI)
        request: Текст запроса
        response: Текст ответа
        error: Сообщение об ошибке (если есть)
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "request": request,
        "response": response,
    }
    if error:
        log_entry["error"] = error

    logger.setLevel(logging.INFO)
    logger.info(f"API Call: {json.dumps(log_entry, ensure_ascii=False)}")
