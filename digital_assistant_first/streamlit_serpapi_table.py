# streamlit_app.py

# Импорты стандартной библиотеки
import logging
import time
import yaml
import asyncio

# Импорты сторонних библиотек
import streamlit as st
from dotenv import load_dotenv

# Локальные импорты
from langchain_openai import ChatOpenAI
from digital_assistant_first.utils.database import generate_csv_from_db
from digital_assistant_first.telegram_system.telegram_data_initializer import update_telegram_messages
from digital_assistant_first import offergen
from digital_assistant_first.interface import (
    init_message_history,
    display_chat_history,
    handle_user_input,
)

def setup_logging():
    """Настройка конфигурации логирования."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def load_config_yaml(config_file="config.yaml"):
    """Загрузить конфигурацию из YAML-файла."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml


def load_available_models():
    """Загрузка доступных моделей из Ollama и добавление пользовательских моделей."""
    models = ["gpt-4o", "gpt-4o-mini"]
    return models


def initialize_session_state(defaults):
    """Инициализировать некоторые значения по умолчанию в st.session_state."""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_configuration():
    """Применить выбранную конфигурацию и обновить состояние сессии."""
    config = {
        "Model": st.session_state["selected_model"],
        "Chain_type": st.session_state["selected_chain_type"],
        "System_type": st.session_state["selected_system"],
        "Temperature": st.session_state["selected_temperature"],
        "Embedding": st.session_state["selected_embedding_model"],
        "Splitter": {
            "Type": st.session_state["selected_splitter_type"],
            "Chunk_size": st.session_state["chunk_size"],
        },
        "history": st.session_state["history"],
        "history_size": st.session_state["history_size"],
        "uploaded_file": st.session_state["uploaded_file"],
        "telegram_enabled": st.session_state["telegram_enabled"],
        "2gis-key": st.session_state["2gis-key"],
        "internet_search": st.session_state["internet_search"],
        "system_prompt": st.session_state["system_prompt"],
        "system_prompt_tickets": st.session_state["system_prompt_tickets"],
    }

    if (
        st.session_state["selected_system"] == "File"
        and st.session_state.get("uploaded_file") is not None
    ):
        config["Uploaded_file"] = st.session_state["uploaded_file"]

    st.session_state["config"] = config
    st.session_state["config_applied"] = True
    time.sleep(2)
    st.rerun()


def initialize_model(config):
    """Инициализация языковой модели на основе конфигурации."""
    return ChatOpenAI(model=config["Model"], stream=True)


def display_banner_and_title():
    """Отображение баннера и заголовка."""
    st.image("https://i.ibb.co/yPcRsgx/AMA.png", use_container_width=True, width=3000)
    st.title("Цифровой Помощник AMA")


def chat_interface(config):
    """
    Основной интерфейс чата:
      1) Инициализируем историю сообщений
      2) Показываем уже накопленную переписку
      3) Обрабатываем новый ввод пользователя
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Конфигурация загружена: {config}")

    template_prompt = "Я ваш Цифровой Ассистент - пожалуйста, задайте свой вопрос."

    # Инициализация модели
    model = initialize_model(config)

    # Инициализация истории, если её нет
    init_message_history(template_prompt)

    # Отображаем историю (включая кнопки лайк/дизлайк для сообщений ассистента)
    display_chat_history()

    # Поле для ввода текста (чат-форма внизу)
    prompt = st.chat_input("Введите запрос здесь...")
    if prompt:
        # Обрабатываем ввод и генерируем ответ
        handle_user_input(model, config, prompt)


def main():
    """Основная функция для запуска приложения Streamlit."""
    load_dotenv()
    logger = setup_logging()
    config_yaml = load_config_yaml()

    defaults = {
        "config_applied": False,
        "config": None,
        "selected_model": config_yaml["model"],
        "selected_system": "RAG",
        "selected_chain_type": "refine",
        "selected_temperature": 0.2,
        "selected_embedding_model": "OpenAIEmbeddings",
        "selected_splitter_type": "character",
        "chunk_size": 2000,
        "history": "On",
        "history_size": 10,
        "uploaded_file": None,
        "telegram_enabled": config_yaml["telegram_enabled"],
        "2gis-key": config_yaml["2gis-key"],
        "internet_search": config_yaml["internet_search"],
        "system_prompt": config_yaml["system_prompt"],
        "system_prompt_tickets": config_yaml["system_prompt_tickets"],
    }

    initialize_session_state(defaults)

    # Сайдбар
    with st.sidebar:
        mode = st.radio("Выберите режим:", ("Чат", "Поиск по картам 2ГИС", "Генерация офферов"))

        # Обновление Telegram-данных при необходимости
        if st.session_state.get("telegram_enabled", False):
            async def initialize_data():
                await update_telegram_messages()
            asyncio.run(initialize_data())

        # Кнопка для скачивания БД
        csv_data = generate_csv_from_db()
        st.download_button(
            label="Скачать БД",
            data=csv_data,
            file_name="chat_history_export.csv",
            mime="text/csv",
        )

    # Применяем конфигурацию (если ещё не применили)
    if not st.session_state["config_applied"]:
        apply_configuration()
    else:
        display_banner_and_title()
        # Устанавливаем режим (Chat / 2Gis / Offers) в конфиг
        if mode == "Поиск по картам 2ГИС":
            st.session_state["config"]["mode"] = "2Gis"
        elif mode == "Генерация офферов":
            st.session_state["config"]["mode"] = "Offers"
        else:
            st.session_state["config"]["mode"] = "Chat"

        # Запускаем общий интерфейс чата
        chat_interface(st.session_state["config"])


if __name__ == "__main__":
    main()