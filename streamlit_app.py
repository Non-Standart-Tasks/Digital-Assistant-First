# Импорты стандартной библиотеки

import logging
import time
import yaml
import asyncio

# Импорты сторонних библиотек
import streamlit as st
from digital_assistant_first.utils.database import init_db 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from digital_assistant_first.interface import *
from langchain_core.documents import Document
from digital_assistant_first.utils.database import generate_csv_from_db
from digital_assistant_first.telegram_system.telegram_data_initializer import (
    update_telegram_messages,
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
        "use_openai_web_search": st.session_state.get("use_openai_web_search", False),
        "web_search_context_size": st.session_state.get("web_search_context_size", "medium"),
        "system_prompt": st.session_state["system_prompt"],
        "system_prompt_aviasales": st.session_state["system_prompt_aviasales"],
        "system_prompt_airport": st.session_state["system_prompt_airport"],
        "system_prompt_tickets": st.session_state["system_prompt_tickets"],
        "offers_enabled": st.session_state["offers_enabled"],
        "FORMAT_INSTRUCTIONS": st.session_state["FORMAT_INSTRUCTIONS"],
        "global_prompt": st.session_state["global_prompt"],
        "system_prompt_tickets_for_aviasales_economy_helper": st.session_state["system_prompt_tickets_for_aviasales_economy_helper"]
    }

    st.session_state["config"] = config
    st.session_state["config_applied"] = True
    time.sleep(2)
    st.rerun()


def display_banner_and_title():
    """Отображение баннера и заголовка."""
    st.image("https://i.ibb.co/yPcRsgx/AMA.png", use_container_width=True, width=3000)
    st.title("Цифровой Помощник AMA")


def chat_interface(config):
    """Отображение интерфейса чата на основе примененной конфигурации."""
    logger = logging.getLogger(__name__)
    logger.info(f"Конфигурация загружена")

    template_prompt = "Я ваш Цифровой Ассистент - пожалуйста, задайте свой вопрос."

    model = initialize_model(config)

    init_message_history(template_prompt)
    display_chat_history()
    prompt = st.chat_input("Введите запрос здесь...")
    if prompt:
        # Используем полностью синхронную версию функции для обработки ввода
        handle_user_input_sync(model, config, prompt)
        st.rerun()


def handle_toggle(key, label, default_value=False):
    """Handle toggle state changes and update config."""
    current_value = st.session_state.get(key, default_value)
    new_value = st.toggle(label, value=current_value)
    
    if new_value != current_value:
        st.session_state[key] = new_value
        if st.session_state.get("config") is not None:
            st.session_state["config"][key] = new_value
            # Remove page rerun to prevent the introduction message from disappearing
            # when toggles are changed
    
    return new_value


def main():
    """Основная функция для запуска приложения Streamlit."""
    init_db()
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
        "use_openai_web_search": config_yaml.get("use_openai_web_search", False),
        "web_search_context_size": config_yaml.get("web_search_context_size", "medium"),
        "system_prompt": config_yaml["system_prompt"],
        "system_prompt_aviasales": config_yaml["system_prompt_aviasales"],
        "system_prompt_airport": config_yaml["system_prompt_airport"],
        "system_prompt_tickets": config_yaml["system_prompt_tickets"],
        "offers_enabled": False,  # По умолчанию офферы отключены
        "maps_2gis_enabled": False,
        "FORMAT_INSTRUCTIONS": config_yaml["FORMAT_INSTRUCTIONS"],
        "global_prompt": config_yaml["global_prompt"],
        "system_prompt_tickets_for_aviasales_economy_helper": config_yaml["system_prompt_tickets_for_aviasales_economy_helper"]
    }

    initialize_session_state(defaults)

    with st.sidebar:
        st.markdown("### Панель управления")
        deepsearch_enabled = handle_toggle("deepsearch", "Включить DeepSearch")
        if deepsearch_enabled != st.session_state.get("deepsearch", False):
            st.session_state["deepsearch"] = deepsearch_enabled
            # Обновить конфигурацию при изменении toggle
            if st.session_state.get("config") is not None:
                st.session_state["config"]["deepsearch"] = deepsearch_enabled
        
        
        new_offers_enabled = handle_toggle("offers_enabled", "Включить офферы")
        if new_offers_enabled != st.session_state.get("offers_enabled", False):
            st.session_state["offers_enabled"] = new_offers_enabled
            # Обновить конфигурацию при изменении toggle
            if st.session_state.get("config") is not None:
                st.session_state["config"]["offers_enabled"] = new_offers_enabled
                # Don't trigger rerun here

        maps_2gis_enabled = handle_toggle("maps_2gis_enabled", "Включить поиск по 2GIS")
        if maps_2gis_enabled != st.session_state.get("maps_2gis_enabled", False):
            st.session_state["maps_2gis_enabled"] = maps_2gis_enabled
            # Обновить конфигурацию при изменении toggle
            if st.session_state.get("config") is not None:
                st.session_state["config"]["maps_2gis_enabled"] = maps_2gis_enabled
                # Don't trigger rerun here
        
        #Пока выключим телеграм функционал
        #if st.session_state.get("telegram_enabled", False):
        #    async def initialize_data():
        #        await update_telegram_messages()
        #    asyncio.run(initialize_data())
        
        csv_data = generate_csv_from_db()
        st.download_button(
            label="Скачать БД",
            data=csv_data,
            file_name="chat_history_export.csv",
            mime="text/csv",
        )

    # Применяем конфигурацию сразу без выбора
    if not st.session_state["config_applied"]:
        apply_configuration()
    else:
        display_banner_and_title()
        st.session_state["config"]["mode"] = "Chat"
        chat_interface(st.session_state["config"])

    
if __name__ == "__main__":
    main()