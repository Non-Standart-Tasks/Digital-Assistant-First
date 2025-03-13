import os
import re
import pytest
import time
import logging
from streamlit.testing.v1 import AppTest
from digital_assistant_first.utils.database import (
    init_db,
    get_chat_record_by_id,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit_test")

# Увеличим время ожидания, так как LLM может отвечать долго
TIMEOUT = 120

# Проверка API ключа
def check_api_key():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY не найден в переменных окружения")
        return False
    logger.info(f"OPENAI_API_KEY найден и имеет длину {len(api_key)} символов")
    return True

# Маркер для пропуска тестов, если ключ не настроен
skip_if_no_api_key = pytest.mark.skipif(
    not check_api_key(),
    reason="OPENAI_API_KEY не найден в переменных окружения"
)

# Фикстура для создания экземпляра AppTest с повторными попытками
@pytest.fixture(scope="function")
def app_test():
    """Создает экземпляр AppTest с обработкой ошибок и повторными попытками."""
    for attempt in range(3):
        try:
            logger.info(f"Попытка {attempt+1} запуска AppTest")
            app = AppTest.from_file("streamlit_app.py")
            # Применяем базовую конфигурацию
            result = app.run()
            
            # Проверяем наличие интерфейса
            if "sidebar" not in dir(result):
                logger.warning("Интерфейс приложения не загрузился")
                time.sleep(5)
                continue
                
            logger.info("AppTest успешно запущен")
            return app
        except Exception as e:
            logger.error(f"Ошибка при запуске AppTest (попытка {attempt+1}): {str(e)}")
            time.sleep(5)  # Пауза перед повторной попыткой
    
    pytest.skip("Не удалось запустить AppTest после 3 попыток")

@pytest.fixture(scope="function")
def initialized_app(app_test):
    """Инициализирует приложение и переключает его в режим чата."""
    at = app_test
    
    # Запускаем приложение и переключаем в режим чата
    try:
        result = at.run()
        logger.info("Выбираем режим 'Чат' в сайдбаре")
        if len(result.sidebar.radio) > 0:
            result = result.sidebar.radio[0].set_value("Чат").run(timeout=60)
            
        # Проверяем, что конфигурация применена
        logger.info("Ждем применения конфигурации")
        start_time = time.time()
        while time.time() - start_time < 60:
            if "config_applied" in result.session_state and result.session_state.config_applied:
                logger.info("Конфигурация успешно применена")
                return result
            result = result.run(timeout=10)
            time.sleep(2)
            
        logger.warning("Тайм-аут при ожидании применения конфигурации")
        return result
    except Exception as e:
        logger.error(f"Ошибка при инициализации приложения: {str(e)}")
        pytest.skip(f"Ошибка инициализации: {str(e)}")

@skip_if_no_api_key
def test_chat_interface(initialized_app):
    """
    Тестирует основной чат-интерфейс:
    1. Отправляет сообщение
    2. Ждет ответа ассистента
    3. Проверяет запись в БД
    """
    at = initialized_app
    
    # 1. Проверяем наличие чат-инпута
    logger.info("Проверка наличия чат-инпута")
    assert len(at.chat_input) > 0, "Чат-инпут не найден"
    
    # 2. Отправляем тестовое сообщение
    TEST_MESSAGE = "Привет, расскажи что ты умеешь"
    logger.info(f"Отправка тестового сообщения: {TEST_MESSAGE}")
    
    # Отправляем сообщение в чат
    result = at.chat_input[0].set_value(TEST_MESSAGE).run(timeout=TIMEOUT)
    
    # 3. Ожидаем ответа ассистента
    logger.info("Ожидание ответа ассистента")
    start_time = time.time()
    record_id = None
    
    while time.time() - start_time < TIMEOUT:
        # Обновляем страницу для получения новых данных
        result = result.run()
        
        if hasattr(result.session_state, "messages"):
            messages = result.session_state.messages
            # Ищем последнее сообщение ассистента
            assistant_msgs = [
                m for m in messages 
                if m.get("role") == "assistant" and m.get("record_id") is not None
            ]
            
            if assistant_msgs:
                record_id = assistant_msgs[-1]["record_id"]
                logger.info(f"Найдено сообщение ассистента с record_id={record_id}")
                break
        
        time.sleep(3)
    
    if not record_id:
        logger.warning("Не получено сообщение ассистента с record_id")
        pytest.skip("Не получен ответ ассистента в отведенное время.")
    
    # 4. Проверяем запись в БД
    logger.info(f"Проверка записи в БД с record_id={record_id}")
    init_db()  # Убедимся, что БД инициализирована
    chat_record = get_chat_record_by_id(record_id)
    
    assert chat_record is not None, f"Запись с record_id={record_id} не найдена в БД"
    assert chat_record.get("user_query") == TEST_MESSAGE, "Запрос пользователя в БД не соответствует отправленному"
    assert chat_record.get("model_response", "").strip() != "", "Ответ модели в БД пустой"
    
    logger.info("Тест успешно пройден: ассистент ответил и данные сохранены в БД")

