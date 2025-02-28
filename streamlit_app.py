# Импорты стандартной библиотеки
import logging
import time

# Импорты сторонних библиотек
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Локальные импорты
from src.interface import *

from src.telegram_system.telegram_data_initializer import update_telegram_messages

def setup_logging():
    """Настройка конфигурации логирования."""
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config_yaml(config_file="config.yaml"):
    """Загрузить конфигурацию из YAML-файла."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml

def load_available_models():
    """Загрузка доступных моделей из Ollama и добавление пользовательских моделей."""
    models = ['gpt-4o', 'gpt-4o-mini']
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
        "system_prompt": st.session_state["system_prompt"],
        "system_prompt_tickets": st.session_state["system_prompt_tickets"]
    }

    if st.session_state["selected_system"] == 'File' and st.session_state.get("uploaded_file") is not None:
        config['Uploaded_file'] = st.session_state["uploaded_file"]

    st.session_state['config'] = config
    st.session_state['config_applied'] = True
    time.sleep(2)
    st.rerun()


def initialize_model(config):
    """Инициализация языковой модели на основе конфигурации."""
    return ChatOpenAI(model=config["Model"], stream=True)



def initialize_vector_store(config):
    """Создание векторного пространства на основе конфигурации."""
    return create_vector_space(config)


def display_banner_and_title():
    """Отображение баннера и заголовка."""
    st.image(
        'https://i.ibb.co/yPcRsgx/AMA.png',
        use_container_width=True, width=3000
    )
    st.title("Цифровой Помощник AMA")


def chat_interface(config):
    """Отображение интерфейса чата на основе примененной конфигурации."""
    logger = logging.getLogger(__name__)
    logger.info(f"Конфигурация загружена: {config}")

    template_prompt = "Я ваш Цифровой Ассистент - пожалуйста, задайте свой вопрос."

    model = initialize_model(config)
    

    init_message_history(template_prompt)
    display_chat_history()
    handle_user_input(model, config)


def main():
    """Основная функция для запуска приложения Streamlit."""
    load_dotenv()
    logger = setup_logging()
    config_yaml = load_config_yaml()
    
    defaults = {
        'config_applied': False,
        'config': None,
        'selected_model': config_yaml['model'],
        'selected_system': 'RAG',
        'selected_chain_type': 'refine',
        'selected_temperature': 0.2,
        'selected_embedding_model': 'OpenAIEmbeddings',
        'selected_splitter_type': 'character',
        'chunk_size': 2000,
        'history': 'On',
        'history_size': 10, 
        'uploaded_file': None,
        'telegram_enabled': config_yaml['telegram_enabled'],
        '2gis-key': config_yaml['2gis-key'],
        'internet_search': config_yaml['internet_search'],
        'system_prompt': config_yaml['system_prompt'],
        'system_prompt_tickets': config_yaml['system_prompt_tickets']

    }
    
    initialize_session_state(defaults)

    mode = st.sidebar.radio("Выберите режим:", ("Чат", "Поиск по картам 2ГИС"))

    if st.session_state.get("telegram_enabled", False):
        async def initialize_data():
            await update_telegram_messages()
        asyncio.run(initialize_data())
    
    # Применяем конфигурацию сразу без выбора
    if not st.session_state['config_applied']:
        apply_configuration()
    else:
        display_banner_and_title()
        if mode == "Поиск по картам 2ГИС":
            st.session_state['config']['mode'] = '2Gis'
            chat_interface(st.session_state['config'])
        else:
            st.session_state['config']['mode'] = 'Chat'
            chat_interface(st.session_state['config'])


if __name__ == "__main__":
    main()
