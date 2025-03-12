import pytest
import sqlite3
import os
import threading
from digital_assistant_first.utils.database import (
    init_db,
    insert_chat_history_return_id,
    update_chat_history_rating_by_id,
    get_chat_record_by_id,
    generate_csv_from_db
)

# Глобальный путь к тестовой БД
TEST_DB_PATH = "test_chat_history.db"

@pytest.fixture(scope="module")
def setup_test_db():
    """Создаем тестовую базу данных с измененным путем"""
    # Сохраняем оригинальный путь
    import digital_assistant_first.utils.database as db_module
    original_path = db_module.DB_PATH
    
    # Меняем путь к БД для тестов вручную
    db_module.DB_PATH = TEST_DB_PATH
    
    # Инициализируем тестовую БД
    db_module.init_db()
    
    yield  # Выполняем тесты
    
    # Восстанавливаем оригинальный путь и удаляем тестовую БД
    db_module.DB_PATH = original_path
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def test_db_initialization(setup_test_db):
    """Тест инициализации базы данных"""
    # Проверяем, что файл БД создан
    assert os.path.exists(TEST_DB_PATH)
    
    # Проверяем, что таблица chat_history существует
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    # Проверяем, что таблицы созданы
    table_names = [table[0] for table in tables]
    assert "chat_history" in table_names
    conn.close()

def test_insert_chat_history(setup_test_db):
    """Тест вставки записи в историю чата"""
    # Вставляем тестовую запись
    user_query = "тестовый запрос"
    model_response = "тестовый ответ"
    mode = "RAG"
    
    record_id = insert_chat_history_return_id(
        user_query=user_query,
        model_response=model_response,
        mode=mode
    )
    
    # Проверяем, что запись добавлена и ID возвращен
    assert record_id > 0
    
    # Получаем запись по ID и проверяем данные
    record = get_chat_record_by_id(record_id)
    assert record is not None
    assert record["user_query"] == user_query
    assert record["model_response"] == model_response
    assert record["mode"] == mode
    assert record["rating"] is None

def test_update_chat_history_rating(setup_test_db):
    """Тест обновления рейтинга в истории чата"""
    # Вставляем тестовую запись
    user_query = "запрос для обновления рейтинга"
    model_response = "ответ для обновления рейтинга"
    mode = "default"
    
    record_id = insert_chat_history_return_id(
        user_query=user_query,
        model_response=model_response,
        mode=mode
    )
    
    # Обновляем рейтинг
    new_rating = "+"
    update_chat_history_rating_by_id(record_id, new_rating)
    
    # Проверяем, что рейтинг обновлен
    record = get_chat_record_by_id(record_id)
    assert record["rating"] == new_rating

def test_get_nonexistent_record(setup_test_db):
    """Тест получения несуществующей записи"""
    # Пытаемся получить запись с несуществующим ID
    record = get_chat_record_by_id(99999)
    assert record is None

def test_generate_csv(setup_test_db):
    """Тест генерации CSV из базы данных"""
    # Вставляем тестовую запись
    user_query = "запрос для CSV"
    model_response = "ответ для CSV"
    mode = "2Gis"
    
    insert_chat_history_return_id(
        user_query=user_query,
        model_response=model_response,
        mode=mode
    )
    
    # Генерируем CSV
    csv_data = generate_csv_from_db()
    
    # Проверяем формат CSV
    assert isinstance(csv_data, str)
    assert "id;user_query;model_response;rating;mode;comment" in csv_data
    assert user_query in csv_data
    assert model_response in csv_data
    assert mode in csv_data

def test_concurrent_db_access(setup_test_db):
    """Тест конкурентного доступа к базе данных"""
    def insert_record(thread_id):
        """Функция для вставки записи из потока"""
        user_query = f"запрос из потока {thread_id}"
        model_response = f"ответ из потока {thread_id}"
        mode = "test"
        
        try:
            record_id = insert_chat_history_return_id(
                user_query=user_query,
                model_response=model_response,
                mode=mode
            )
            return record_id
        except Exception as e:
            return None
    
    # Создаем несколько потоков для конкурентной записи
    threads = []
    record_ids = []
    for i in range(5):
        thread = threading.Thread(target=lambda i=i: record_ids.append(insert_record(i)))
        threads.append(thread)
        thread.start()
    
    # Ждем завершения всех потоков
    for thread in threads:
        thread.join()
    
    # Проверяем, что все записи были успешно добавлены
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE mode = ?", ("test",))
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count == 5
    assert len(record_ids) == 5
    assert all(id is not None for id in record_ids)

def test_db_error_handling(setup_test_db):
    """Тест обработки ошибок базы данных"""
    # Пробуем выполнить некорректный запрос к базе напрямую
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    
    with pytest.raises(sqlite3.OperationalError):
        cursor.execute("SELECT * FROM non_existent_table")
    
    # Проверяем, что база данных все еще доступна после ошибки
    cursor.execute("SELECT COUNT(*) FROM chat_history")
    count = cursor.fetchone()[0]
    assert isinstance(count, int)
    conn.close()

def test_multiple_updates(setup_test_db):
    """Тест нескольких последовательных обновлений рейтинга"""
    # Вставляем тестовую запись
    record_id = insert_chat_history_return_id(
        user_query="запрос для множественных обновлений",
        model_response="ответ для множественных обновлений",
        mode="default"
    )
    
    # Обновляем рейтинг несколько раз
    ratings = ["+", "-", None, "+"]
    
    for rating in ratings:
        update_chat_history_rating_by_id(record_id, rating)
        record = get_chat_record_by_id(record_id)
        assert record["rating"] == rating