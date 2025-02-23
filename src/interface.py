#Импорты стандартной библиотеки
import logging
import json
import tempfile
import pymupdf
import os
import asyncio
import yaml
import pandas as pd
import streamlit as st
from html import escape

# Импорты сторонних библиотек
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.utils.check_serp_response import APIKeyManager

from src.utils.logging import setup_logging, log_api_call
from src.internet_search import *

import requests
import pydeck as pdk

# Локальные импорты
from src.utils.kv_faiss import KeyValueFAISS
from src.utils.paths import ROOT_DIR
from src.telegram_system.telegram_rag import EnhancedRAGSystem
from src.telegram_system.telegram_data_initializer import update_telegram_messages
from src.telegram_system.telegram_data_initializer import TelegramManager
from src.telegram_system.telegram_initialization import fetch_telegram_data
from src.utils.aviasales_parser import fetch_page_text, construct_aviasales_url
from src.geo_system.two_gis import fetch_2gis_data
from src.websites_rag.yndx_restaurants import (
    analyze_restaurant_request,
    get_restaurants_by_category,
    fetch_yndx_context
)

logger = setup_logging(logging_path='logs/digital_assistant.log')

serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.csv")

def aviasales_request(model, config, user_input):
    # Вызываем модель с параметром stream=False
    messages = [
                {"role": "system", "content": config['system_prompt_tickets']},
                {"role": "user", "content": user_input}
                ]
    
    response = model.invoke(
        messages,
        stream=False
    )

    # Получаем контент из ответа
    if hasattr(response, 'content'):
        content = response.content
    elif hasattr(response, 'message'):
        content = response.message.content
    else:
        content = str(response)

    analysis = content.strip()
    if analysis.startswith("```json"):
        analysis = analysis[7:]  # Remove ```json
    if analysis.endswith("```"):
        analysis = analysis[:-3]  # Remove trailing ```
    analysis = analysis.strip()
    tickets_need = json.loads(analysis)
    return tickets_need


def model_response_generator(model, config):
    """Сгенерировать ответ с использованием модели и ретривера."""
    # Получение последнего пользовательского ввода
    user_input = st.session_state["messages"][-1]["content"]

    # Получение информации о билетах и контекста ресторанов
    tickets_need = aviasales_request(model, config, user_input)
    restaurant_context_text = fetch_yndx_context(user_input, model)

    try:
        # Формирование истории сообщений (без системных сообщений)
        message_history = ""
        if "messages" in st.session_state and len(st.session_state["messages"]) > 1:
            history_messages = [
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state["messages"]
                if msg.get("role") != "system"
            ]
            history_size = int(config.get("history_size", 0))
            if history_size:
                history_messages = history_messages[-history_size:]
            message_history = "\n".join(history_messages)

        # Если включён интернет-поиск, выполнить его; иначе, оставить пустые значения
        if config.get("internet_search", False):
            _, serpapi_key = serpapi_key_manager.get_best_api_key()
            shopping_res = search_shopping(user_input, serpapi_key)
            internet_res, links, coordinates = search_places(user_input, serpapi_key)
            maps_res = search_map(user_input, coordinates, serpapi_key)
            # yandex_res = yandex_search(user_input, serpapi_key)
        else:
            shopping_res = ""
            internet_res = ""
            links = ""
            maps_res = ""
            # yandex_res = ""

        # Если требуется, сформировать URL для Aviasales
        if tickets_need.get("response", "").lower() == "true":
            aviasales_url = construct_aviasales_url(
                tickets_need["departure_city"],
                tickets_need["destination"],
                tickets_need["start_date"],
                tickets_need["end_date"],
                tickets_need["passengers"],
                tickets_need.get("travel_class", ""),
            )
        else:
            aviasales_url = ""

        # Если включена поддержка Telegram, получить контекст из Telegram
        if config.get("telegram_enabled", False):
            telegram_manager = TelegramManager()
            rag_system = EnhancedRAGSystem(
                data_file="data/telegram_messages.json",
                index_directory="data/"
            )
            telegram_context = fetch_telegram_data(user_input, rag_system, k=50)
        else:
            telegram_context = ""

        # Формирование запроса для ресторанного контекста, если он есть
        restaurants_prompt = restaurant_context_text if restaurant_context_text else ""

        # Загрузка шаблона системного промпта из конфигурации
        system_prompt_template = config["system_prompt"]

        # Форматирование системного промпта с подстановкой переменных
        formatted_prompt = system_prompt_template.format(
            context=message_history,
            internet_res=internet_res,
            links=links,
            shopping_res=shopping_res,
            maps_res=maps_res,
            # yandex_res=yandex_res,
            telegram_context=telegram_context,
            yndx_restaurants=restaurants_prompt
        )

        # Если режим '2Gis', получить дополнительные данные для таблицы и карты
        table_data = []
        pydeck_data = []
        if config.get("mode") == "2Gis":
            table_data, pydeck_data = fetch_2gis_data(user_input, config)

        # Формирование шаблона сообщений для запроса с помощью ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_prompt),
            ("human", "User query: {input}\nAdditional context: {context}")
        ])
        # Форматирование сообщений, подставляя входные данные пользователя (дополнительного контекста нет)
        messages = prompt.format(input=user_input, context="")

        # Вызов модели с потоковой передачей ответа
        response = model.invoke(messages, stream=True)

        # Извлечение ответа из модели (поддержка разных вариантов формата ответа)
        if hasattr(response, "content"):
            answer = response.content
        elif hasattr(response, "message"):
            answer = response.message.content
        else:
            answer = str(response)

        # Если данных для таблицы и pydeck нет, оставить пустыми списками
        table_data = table_data if table_data else []
        pydeck_data = pydeck_data if pydeck_data else []

        # Возврат результата вместе с дополнительными данными
        yield {
            "answer": answer,
            "maps_res": maps_res,
            "aviasales_link": aviasales_url,
            "table_data": table_data,
            "pydeck_data": pydeck_data
        }

        # Логирование вызова API
        log_api_call(
            logger=logger,
            source=f"LLM ({config['Model']})",
            request=user_input,
            response=answer,
        )

    except Exception as e:
        # Логирование ошибки при вызове API
        log_api_call(
            logger=logger,
            source=f"LLM ({config['Model']})",
            request=user_input,
            response="",
            error=str(e)
        )
        raise
def handle_user_input(model, config):
    """Обработать пользовательский ввод и сгенерировать ответ ассистента."""
    prompt = st.chat_input("Введите запрос здесь...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            maps_res = []  # Инициализируем maps_res
            for chunk in model_response_generator(model, config):
                response_text += chunk["answer"]

                # Проверяем наличие ключа aviasales_link
                if "aviasales_link" in chunk:
                    aviasales_link = chunk["aviasales_link"]
                    # Если значение непустое, добавляем с префиксом, иначе просто добавляем его (обычно пустое)
                    if aviasales_link and aviasales_link.strip():
                        response_text += f"\n\n### Данные из Авиасейлс \n **Ссылка** - {aviasales_link}"
                    else:
                        response_text += f"\n\n{aviasales_link}"
                
                if config['mode'] == '2Gis':
                    
                    response_text += f"\n\n### Данные из 2Гис"
                    if 'table_data' in chunk:
                        df = pd.DataFrame(chunk['table_data'])
                        st.dataframe(df)  # Красивое представление таблицы
                    else:
                        st.warning("Ничего не найдено.")

                    # Отрисовка PyDeck карты
                    if 'pydeck_data' in chunk:
                        df_pydeck = pd.DataFrame(chunk['pydeck_data'])
                        st.subheader("Карта")
                        st.pydeck_chart(
                            pdk.Deck(
                                map_style=None,
                                initial_view_state=pdk.ViewState(
                                    latitude=df_pydeck["lat"].mean(),
                                    longitude=df_pydeck["lon"].mean(),
                                    zoom=13
                                ),
                                layers=[
                                    pdk.Layer(
                                        "ScatterplotLayer",
                                        data=df_pydeck,
                                        get_position="[lon, lat]",
                                        get_radius=30,
                                        get_fill_color=[255, 0, 0],
                                        pickable=True
                                    )
                                ],
                                tooltip={
                                    "html": "<b>{name}</b>",
                                    "style": {
                                        "color": "white"
                                    }
                                }
                            )
                        )
                    else:
                        st.warning("Не найдено точек для отображения на PyDeck-карте.")
                            
                    response_placeholder.markdown(response_text)
                
                    if isinstance(chunk.get("maps_res"), list):
                        maps_res = chunk["maps_res"]


                response_placeholder.markdown(response_text)
                
                if isinstance(chunk.get("maps_res"), list):
                    maps_res = chunk["maps_res"]

            st.session_state["messages"].append(
                {"role": "assistant", "content": response_text, "question": prompt}
            )

               # Проверка и обработка maps_res

     
        
def init_message_history(template_prompt):
    """Инициализировать историю сообщений для чата."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        with st.chat_message('System'):
            st.markdown(template_prompt)
        


def display_chat_history():
    """Отобразить историю чата из состояния сессии."""
    for message in st.session_state["messages"][1:]:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            st.markdown(message['content'])


 