import sqlite3
from typing import Optional
import io
import csv

# Если у вас есть отдельный модуль paths.py с ROOT_DIR,
# замените строку ниже на:
# from digital_assistant_first.utils.paths import ROOT_DIR
# DB_PATH = f"{ROOT_DIR}/chat_history.db"
DB_PATH = "chat_history.db"


def init_db():
    """Инициализирует таблицу chat_history, если её ещё нет."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_query TEXT,
                model_response TEXT,
                rating TEXT,
                mode TEXT
            )
            """
        )
        conn.commit()


def insert_chat_history_return_id(
    user_query: str,
    model_response: str,
    mode: str,
    rating: str = None
) -> int:
    """
    Вставляет запись в chat_history и возвращает её ID.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_history (user_query, model_response, rating, mode)
            VALUES (?, ?, ?, ?)
            """,
            (user_query, model_response, rating, mode)
        )
        conn.commit()
        return cursor.lastrowid  # ID вставленной строки


def update_chat_history_rating_by_id(record_id: int, new_rating: str):
    """
    Обновляет поле rating у записи с указанным ID.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE chat_history
               SET rating = ?
             WHERE id = ?
            """,
            (new_rating, record_id)
        )
        conn.commit()


def get_chat_record_by_id(record_id: int) -> Optional[dict]:
    """
    Получает запись по ID и возвращает её в виде словаря (или None, если записи нет).
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_query, model_response, rating, mode
              FROM chat_history
             WHERE id = ?
            """,
            (record_id,)
        )
        row = cursor.fetchone()

    if row:
        return {
            "id": row[0],
            "user_query": row[1],
            "model_response": row[2],
            "rating": row[3],
            "mode": row[4]
        }
    return None


def generate_csv_from_db() -> str:
    """
    Возвращает всю таблицу chat_history в формате CSV (одна строка).
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=';')

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, user_query, model_response, rating, mode
            FROM chat_history
        """)
        rows = cursor.fetchall()

    # Пишем заголовки
    writer.writerow(["id", "user_query", "model_response", "rating", "mode"])

    # Пишем строки таблицы
    for row in rows:
        writer.writerow(row)

    csv_data = buffer.getvalue()
    buffer.close()
    return csv_data