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
from src.utils.aviasales_parser import analyze_aviasales_url, construct_aviasales_url, aviasales_request
from src.geo_system.two_gis import fetch_2gis_data
from src.utils.yndx_restaurants import (
    analyze_restaurant_request,
    get_restaurants_by_category,
)

logger = setup_logging(logging_path='logs/digital_assistant.log')

serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.csv")

def fetch_yndx_context(user_input, model):
    restaurant_analysis = analyze_restaurant_request(user_input, model)
    restaurant_context_text = ""
    restaurants_data = []
    if restaurant_analysis.get("restaurant_recommendation", "false").lower() == "true":
        requested_category = restaurant_analysis.get("category", "")
        if requested_category:
            restaurants_data = get_restaurants_by_category(requested_category)
            if restaurants_data:
                # Формируем текстовый блок с информацией о найденных ресторанах
                restaurant_context_parts = []
                
                for r in restaurants_data:
                    restaurant_context_parts.append(
                        f"Название: {r.get('name')}\n"
                        f"Режим работы: {r.get('working_hours')}\n"
                        f"Адрес: {r.get('address', {}).get('street')}\n"
                        f"Метро: {', '.join(r.get('address', {}).get('metro', []))}\n"
                        f"Описание: {r.get('description')}\n"
                        f"Категории: {', '.join(r.get('categories', []))}"
                    )
                restaurant_context_text = "\n\n".join(restaurant_context_parts)

    return restaurant_context_text


def model_response_generator(model, config):
    """Сгенерировать ответ с использованием модели и ретривера."""
    
    user_input = st.session_state["messages"][-1]["content"]
    
    tickets_need = aviasales_request(model, config, user_input)

    restaurant_context_text = fetch_yndx_context(user_input, model)


    try:
        # Формирование истории сообщений (исключая системное сообщение)
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
        
        # Если интернет-поиск включён, вызываем функции поиска, иначе возвращаем пустую строку
        if config.get("internet_search", False):
            _, serpapi_key = serpapi_key_manager.get_best_api_key()

            shopping_res = search_shopping(user_input, serpapi_key)
            internet_res, links, coordinates = search_places(user_input, serpapi_key)
            maps_res = search_map(user_input, coordinates, serpapi_key)
            #yandex_res = yandex_search(user_input, serpapi_key)
        else:
            shopping_res = ""
            internet_res = ""
            links = ""
            maps_res = ""
            #yandex_res = ""

        if tickets_need.get('response', '').lower() == 'true':
            # Get flight options
            aviasales_url = construct_aviasales_url(
                from_city=tickets_need["departure_city"],
                to_city=tickets_need["destination"],
                depart_date=tickets_need["start_date"],
                return_date=tickets_need["end_date"],
                adult_passengers=tickets_need["adult_passengers"],
                child_passengers=tickets_need["child_passengers"],
                travel_class=tickets_need.get("travel_class", ""),
            )
            try:
                avia_result = analyze_aviasales_url(webpage_url=aviasales_url)
            except ConnectionResetError:
                try:
                    avia_result = analyze_aviasales_url(webpage_url=aviasales_url)
                except ConnectionResetError:
                    avia_result = False
            except Exception as e:
                avia_result = False
        else:
            aviasales_url = ''
            avia_result = False
        
        if config.get("telegram_enabled", False):
            telegram_manager = TelegramManager()
            rag_system = EnhancedRAGSystem(
                data_file="data/telegram_messages.json",
                index_directory="data/"
            )

            telegram_context = fetch_telegram_data(user_input, rag_system, k=50)
        else:
            telegram_context = ''

        if restaurant_context_text:
            restaurants_prompt = f"{restaurant_context_text}"
        else:
            restaurants_prompt = ""

        # Загрузка системного промпта из YAML-конфига
        system_prompt_template = config["system_prompt"]
        
        # Форматирование промпта с подстановкой переменных
        formatted_prompt = system_prompt_template.format(
            context=message_history,
            internet_res=internet_res,
            links=links,
            shopping_res=shopping_res,
            maps_res=maps_res,
            #yandex_res=yandex_res,
            telegram_context=telegram_context,
            yndx_restaurants=restaurants_prompt
        )
        # Создание цепочки для модели, если имя модели начинается с 'gpt'

        table_data = []
        pydeck_data = []
        if config.get('mode') == '2Gis':
            table_data, pydeck_data = fetch_2gis_data(user_input, config)


        
        # Формируем шаблон сообщений для запроса
        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_prompt),
            ("human", "User query: {input}\nAdditional context: {context}")
        ])
        
        # Форматируем сообщения, подставляя входные данные пользователя
        messages = prompt.format(input=user_input, context="")  # Если дополнительного контекста нет, можно оставить пустую строку
        
        # Вызываем модель напрямую без retrieval chain
        response = model.invoke(messages, stream=True)
        
        # Извлекаем ответ из модели (поддержка разных вариантов формата ответа)
        if hasattr(response, 'content'):
            answer = response.content
        elif hasattr(response, 'message'):
            answer = response.message.content
        else:
            answer = str(response)
        
        table_data = table_data if table_data else []
        pydeck_data = pydeck_data if pydeck_data else []
        # Выводим ответ вместе с дополнительными данными (например, maps_res)
        yield {
            "answer": answer,
            "maps_res": maps_res,
            "aviasales_link": aviasales_url,
            "aviasales_screen": avia_result,
            "table_data": table_data,
            "pydeck_data": pydeck_data
        }

        
        log_api_call(
                logger=logger,
                source=f"LLM ({config['Model']})",
                request=user_input,
                response=answer,
            )

    except Exception as e:
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

                if "aviasales_screen" in chunk:
                    if chunk['aviasales_screen']:
                        st.image('screenshot.png')
                
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


def creating_documents(config):
    """Создать векторное пространство на основе конфигурации."""
    if config["System_type"] == 'File' and 'Uploaded_file' in config and config['Uploaded_file'] is not None:
        uploaded_file = config['Uploaded_file']
        # Обработка загруженного файла
        if uploaded_file.type == "text/plain":
            # Чтение текстового файла
            file_content = uploaded_file.getvalue().decode("utf-8")
            documents = [Document(page_content=file_content)]
            return documents
        
        elif uploaded_file.type == "application/pdf":
       # Чтение PDF файла с использованием PyMuPDF
            # Сохранение загруженного файла во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Используем PyMuPDF для извлечения текста из PDF
            with pymupdf.open(os.path.normpath(tmp_file_path)) as doc:
                text = chr(12).join([page.get_text() for page in doc])
                paragraphs = text.split('\x0c')
                formatted_paragraphs = [
                    ' '.join(paragraph.replace('\n', ' ').replace('\xa0', ' ').split())
                    for paragraph in paragraphs
                ]
                file_content = '\n\n'.join(formatted_paragraphs)
                documents = [Document(page_content=file_content)]

            # Удаление временного файла
            os.unlink(tmp_file_path)
            return documents

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Чтение DOCX файла
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_file_path)
            documents = loader.load()
            return documents
        else:
            st.error("Неподдерживаемый тип файла.")
            return None
    else:
        path = ROOT_DIR / 'content' / 'json' / 'pp_data.json'
        loader = TextLoader(path)
        documents = loader.load()
        
        return documents
    
def create_retriever(splitter_type, embeddings_model, documents, chunk_size):
    """Создать ретривер для извлечения данных."""
    if splitter_type == 'character':
        embeddings = (
            HuggingFaceEmbeddings(model_name=embeddings_model)
            if embeddings_model != 'OpenAIEmbeddings'
            else OpenAIEmbeddings(model='text-embedding-ada-002')
        )
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        docs = splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, embeddings)
    
        return vector_store
    
    elif splitter_type == 'json':
        embeddings = (
            HuggingFaceEmbeddings(model_name=embeddings_model)
            if embeddings_model != 'OpenAIEmbeddings'
            else OpenAIEmbeddings(model='text-embedding-ada-002')
        )
        logical_chunks = json.loads(documents[0].page_content)
        kv_dict = {k: f'{k}:\n{v}' for k, v in logical_chunks.items()}
        docs = [Document(k) for k in logical_chunks.keys()]
        vector_store = KeyValueFAISS.from_documents(docs, embeddings).add_value_documents(kv_dict)
        return vector_store

    else:
        raise ValueError(f"Неподдерживаемый тип разбиения: {splitter_type}")

def create_vector_space(config):
    """Создать векторное пространство для извлечения документов на основе конфигурации."""

    embeddings_model = config['Embedding']
    splitter_type = config['Splitter']['Type']
    chunk_size = int(config['Splitter']['Chunk_size'])

    
    documents = creating_documents(config)

    vector_store = create_retriever(splitter_type, embeddings_model, documents, chunk_size)
    
    return vector_store


 