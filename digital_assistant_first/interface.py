# Импорты стандартной библиотеки
import logging
import json
import asyncio
import pandas as pd
import streamlit as st

# Импорты сторонних библиотек
from langchain_core.prompts import ChatPromptTemplate
from digital_assistant_first.utils.check_serp_response import APIKeyManager
from digital_assistant_first.utils.logging import setup_logging, log_api_call
from digital_assistant_first.internet_search import search_shopping, search_places, yandex_search
import pydeck as pdk
from langchain_openai import ChatOpenAI

# Локальные импорты
from digital_assistant_first.telegram_system.telegram_rag import EnhancedRAGSystem
from digital_assistant_first.telegram_system.telegram_data_initializer import (
    TelegramManager,
)
from digital_assistant_first.telegram_system.telegram_initialization import (
    fetch_telegram_data,
)
from digital_assistant_first.utils.aviasales_parser import AviasalesHandler
from digital_assistant_first.geo_system.two_gis import fetch_2gis_data
from digital_assistant_first.offergen.agent import validation_agent
from digital_assistant_first.offergen.utils import get_system_prompt_for_offers
from digital_assistant_first.yndx_system.restaurant_context import fetch_yndx_context
from digital_assistant_first.utils.link_checker import link_checker, corrector
from digital_assistant_first.utils.database import (
    init_db, 
    insert_chat_history_return_id, 
    update_chat_history_rating_by_id, 
    get_chat_record_by_id
)

from dotenv import load_dotenv

load_dotenv()

logger = setup_logging(logging_path="logs/digital_assistant.log")
serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.csv")
init_db()

# Add the initialize_model function here to avoid circular import
def initialize_model(config):
    """Инициализация языковой модели на основе конфигурации."""
    return ChatOpenAI(model=config["Model"], stream=True)

async def model_response_generator(model, config):
    """Сгенерировать ответ с использованием модели и ретривера асинхронно."""
    user_input = st.session_state["messages"][-1]["content"]
    
    # Подготовка message_history
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

    # Определение категории запроса с помощью агента
    async def categorize_request():
        category_prompt = """
        Определи категорию запроса пользователя и верни ТОЛЬКО одну из следующих категорий без дополнительных пояснений:
        - рестораны (если запрос о ресторанах, кафе, еде, доставке питания и т.п.)
        - ивенты (если запрос о мероприятиях, концертах, выставках, фестивалях и т.п.)
        - поездки (если запрос о поездках на машинах, такси, аренде автомобилей и т.п.)
        - другое (если запрос не подходит ни под одну из перечисленных категорий)
        
        Запрос пользователя: {user_input}
        """
        
        messages = [
            {"role": "system", "content": category_prompt.format(user_input=user_input)}
        ]
        
        response = model.invoke(messages, stream=False)
        
        if hasattr(response, "content"):
            category = response.content.strip().lower()
        elif hasattr(response, "message"):
            category = response.message.content.strip().lower()
        else:
            category = str(response).strip().lower()
        
        # Логирование определенной категории
        logger.info(f"Определена категория запроса: {category}")
        
        return category
    
    # Получаем категорию запроса
    request_category = await categorize_request()
    
    # Создаем список задач для параллельного выполнения
    tasks = []
    
    # Задача для Aviasales (только для категории "поездки" или если это не специфический запрос)
    if request_category == "поездки" or request_category == "другое":
        aviasales_tool = AviasalesHandler()
        tasks.append(aviasales_tool.aviasales_request(model, config, user_input))
    
    # Инициализируем переменные по умолчанию
    shopping_res = ""
    internet_res = ""
    links = ""
    yandex_res = ""
    telegram_context = ""
    table_data = []
    pydeck_data = []
    
    # Задачи для интернет-поиска (всегда выполняем, но используем информацию о категории)
    if config.get("internet_search", False):
        async def fetch_internet_data():
            _, serpapi_key = serpapi_key_manager.get_best_api_key()
            
            # Добавляем информацию о категории к запросу для более точного поиска
            enhanced_query = user_input
            if request_category != "другое":
                enhanced_query = f"{user_input} {request_category}"
                
            shopping = await search_shopping(enhanced_query, serpapi_key)
            internet, links_data, _ = await search_places(enhanced_query, serpapi_key)
            yandex_res = await yandex_search(enhanced_query, serpapi_key)
            return shopping, internet, links_data, yandex_res
        
        tasks.append(fetch_internet_data())
    
    # Задача для Telegram (всегда выполняем)
    if config.get("telegram_enabled", False):
        async def fetch_telegram_data_async():
            telegram_manager = TelegramManager()
            rag_system = EnhancedRAGSystem(
                data_file="data/telegram_messages.json", index_directory="data/"
            )
            return await fetch_telegram_data(user_input, rag_system, k=50)
        
        tasks.append(fetch_telegram_data_async())
    
    # Задача для 2Gis (только для категорий "рестораны" и "ивенты")
    if request_category in ["рестораны", "ивенты"]:
        tasks.append(fetch_2gis_data(user_input, config))
    
    try:
        # Выполняем все задачи параллельно
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обрабатываем результаты
        result_index = 0
        
        # Результат Aviasales
        tickets_need = {"response": "false"}
        if request_category == "поездки" or request_category == "другое":
            if result_index < len(results):
                tickets_need = results[result_index] if not isinstance(results[result_index], Exception) else {"response": "false"}
                result_index += 1
        
        # Результаты интернет-поиска
        if config.get("internet_search", False):
            if not isinstance(results[result_index], Exception):
                shopping_res, internet_res, links, yandex_res = results[result_index]
            result_index += 1
        
        # Результаты Telegram
        if config.get("telegram_enabled", False):
            if not isinstance(results[result_index], Exception):
                telegram_context = results[result_index]
            result_index += 1
        
        # Результаты 2Gis
        if request_category in ["рестораны", "ивенты"]:
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                table_data, pydeck_data = results[result_index]
            result_index += 1
        
        # Формируем URL для Aviasales
        aviasales_url = ""
        aviasales_flight_info = ""
        
        if tickets_need.get("response", "").lower() == "true":
            # Создаем инструмент Aviasales, если он еще не создан
            if 'aviasales_tool' not in locals():
                aviasales_tool = AviasalesHandler()
                
            aviasales_url = aviasales_tool.construct_aviasales_url(
                tickets_need["departure_city"],
                tickets_need["destination"],
                tickets_need["start_date"],
                tickets_need["end_date"],
                tickets_need.get("adult_passengers", 1),
                tickets_need.get("child_passengers", 0),
                tickets_need.get("travel_class", ""),
            )
            if config.get("aviasales_search") == "True":
                aviasales_flight_info = await aviasales_tool.get_info_aviasales_url(aviasales_url=aviasales_url, user_input=user_input)
        else:
            aviasales_flight_info = ""
            
        # Формируем системный промпт
        system_prompt_template = config["system_prompt"]
        
        # Создаем информацию о категории запроса
        category_info = f"Категория запроса пользователя: {request_category}"
        
        formatted_prompt = system_prompt_template.format(
            context=message_history,
            internet_res=internet_res,
            yandex_res=yandex_res,
            links=links,
            shopping_res=shopping_res,
            telegram_context=telegram_context,
            # yndx_restaurants=restaurants_prompt,
            aviasales_flight_info=aviasales_flight_info,
        )
        
        # Добавляем информацию о категории в начало промпта
        formatted_prompt = f"{category_info}\n\n{formatted_prompt}"
        
        # Получаем ответ от модели
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", formatted_prompt),
                ("human", "User query: {input}\nAdditional context: {context}"),
            ]
        )
        messages = prompt_template.format(input=user_input, context="")
        
        # Используем неасинхронную версию с явно отключенным streaming
        response = model.invoke(messages, stream=False)
        
        if hasattr(response, "content"):
            answer = response.content
        elif hasattr(response, "message"):
            answer = response.message.content
        else:
            answer = str(response)
        
        # Проверка и коррекция ссылок
        link_statuses = await link_checker.run(answer)
        
        if link_statuses.data.links:
            some_link_is_invalid = any(
                not link.status for link in link_statuses.data.links
            )
            if some_link_is_invalid:
                corrected_answer = await corrector.run(answer, deps=link_statuses.data.links)
                answer = corrected_answer.data
        
        log_api_call(
            logger=logger,
            source=f"LLM ({config['Model']})",
            request=user_input,
            response=answer,
        )
        
        return {
            "answer": answer,
            "aviasales_link": aviasales_url,
            "table_data": table_data or [],
            "pydeck_data": pydeck_data or [],
            "request_category": request_category,
        }
        
    except Exception as e:
        logger.error(f"Error in model_response_generator: {str(e)}", exc_info=True)
        log_api_call(
            logger=logger,
            source=f"LLM ({config['Model']})",
            request=user_input,
            response="",
            error=str(e),
        )
        raise

def offers_mode_interface(config):
    """
    Режим генерации офферов. Не используем nest_asyncio,
    вручную создаём event loop для вызова асинхронного validation_agent.
    """
    st.subheader("Режим генерации офферов VTB Family")
    if "messages_offers" not in st.session_state:
        st.session_state["messages_offers"] = []
    user_input = st.chat_input("Введите запрос для генерации офферов...")
    if user_input:
        st.session_state["messages_offers"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            validation_result = loop.run_until_complete(
                validation_agent.run(user_input)
            ).data
            if validation_result.number_of_offers_to_generate < 1:
                validation_result.number_of_offers_to_generate = 10
            system_prompt = get_system_prompt_for_offers(validation_result, user_input)
            if system_prompt == "No relevant offers were found for the search request.":
                with st.chat_message("assistant"):
                    st.warning("Офферы не найдены, попробуйте уточнить запрос.")
                return
            model = initialize_model(config)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
            response = model.invoke(messages, stream=True)
            response_text = ""
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                for chunk in response:
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        key, value = chunk
                        if key == "content":
                            response_text += value
                            response_placeholder.markdown(response_text)
        except Exception as e:
            st.error(f"Error in offers generation: {str(e)}")

async def handle_user_input(model, config, prompt):
    """Обработать пользовательский ввод и сгенерировать ответ ассистента."""
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            response = await model_response_generator(model, config)
            response_text += response["answer"]

            # Проверяем наличие ключа aviasales_link
            if "aviasales_link" in response:
                aviasales_link = response["aviasales_link"]
                # Если значение непустое, добавляем с префиксом, иначе просто добавляем его (обычно пустое)
                if aviasales_link and aviasales_link.strip():
                    response_text += f"\n\n### Данные из Авиасейлс \n **Ссылка** - {aviasales_link}"
                else:
                    response_text += f"\n\n{aviasales_link}"
            
            # Получаем категорию запроса из ответа, если она есть
            if response.get("request_category") in ["рестораны", "ивенты"]:
                # Добавляем данные 2GIS для категорий "рестораны" и "ивенты"
                response_text += f"\n\n### Данные из 2Гис"
                if "table_data" in response and response["table_data"]:
                    df = pd.DataFrame(response["table_data"])
                    st.dataframe(df)  # Красивое представление таблицы
                else:
                    st.warning("Ничего не найдено.")

                if "pydeck_data" in response and response["pydeck_data"]:
                    st.session_state["last_pydeck_data"] = response["pydeck_data"]
                    st.session_state["show_map"] = True
                else:
                    st.session_state["last_pydeck_data"] = []
                    st.session_state["show_map"] = False
                    st.warning("Не найдено точек для отображения на PyDeck-карте.")
        
            # Update the response placeholder for each chunk, regardless of mode
            response_placeholder.markdown(response_text)

            st.session_state["messages"].append(
                {
                    "role": "assistant", 
                    "content": response_text, 
                    "question": prompt,
                    "show_map": st.session_state.get("show_map", False),
                    "request_category": response.get("request_category", "")
                }
            )
            
            st.markdown("### Оцените ответ:")
            col1, col2 = st.columns(2)
            if col1.button("👍", key=f"thumbs_up_{len(st.session_state['messages'])}"):
                st.success("Вы поставили 👍")
            if col2.button("👎", key=f"thumbs_down_{len(st.session_state['messages'])}"):
                st.error("Вы поставили 👎")  

            record_id = insert_chat_history_return_id(
            user_query=prompt,
            model_response=response_text,
            mode=config["mode"],
            rating=None
            )

            # В самом сообщении ассистента также сохраним record_id для возможности лайка/дизлайка
            st.session_state["messages"][-1]["record_id"] = record_id

def init_message_history(template_prompt):
    """Инициализировать историю сообщений для чата."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        with st.chat_message("System"):
            st.markdown(template_prompt)


def display_chat_history():
    """Отобразить историю чата из состояния сессии (включая кнопки рейтинга для ассистента)."""
    last_assistant_index = -1
    for i, message in enumerate(st.session_state["messages"]):
        if message["role"] == "assistant":
            last_assistant_index = i
    for i, message in enumerate(st.session_state["messages"]):
        with st.chat_message(message["role"]):
            # Если было сохранено "question", покажем его как заголовок (опционально)
            if "question" in message:
                st.markdown(f"**Вопрос**: {message['question']}")

            # Основной контент сообщения
            st.markdown(message["content"])
            
            # Если это ассистент, обрабатываем рейтинги и карты
            if message["role"] == "assistant":
                # Показываем карту для последнего сообщения ассистента, если категория "рестораны" или "ивенты"
                is_map_needed = (message.get("request_category") in ["рестораны", "ивенты"] or 
                                message.get("show_map", False)) and i == last_assistant_index
                
                if is_map_needed:
                    if "last_pydeck_data" in st.session_state:
                        pydeck_data = st.session_state["last_pydeck_data"]
                        if pydeck_data and len(pydeck_data) > 0:
                            df_pydeck = pd.DataFrame(pydeck_data)
                            st.subheader("Карта")
                            st.pydeck_chart(
                                pdk.Deck(
                                    map_style=None,
                                    initial_view_state=pdk.ViewState(
                                        latitude=df_pydeck["lat"].mean(),
                                        longitude=df_pydeck["lon"].mean(),
                                        zoom=13,
                                    ),
                                    layers=[
                                        pdk.Layer(
                                            "ScatterplotLayer",
                                            data=df_pydeck,
                                            get_position="[lon, lat]",
                                            get_radius=30,
                                            get_fill_color=[255, 0, 0],
                                            pickable=True,
                                        )
                                    ],
                                    tooltip={
                                        "html": "<b>{name}</b>",
                                        "style": {"color": "white"},
                                    },
                                )
                            )
                
                # Показываем кнопки оценки
                record_id = message.get("record_id")
                if record_id:
                    col1, col2 = st.columns(2)

                    if col1.button("👍", key=f"thumbs_up_{i}"):
                        update_chat_history_rating_by_id(record_id, "+")
                        st.session_state["last_rating_action"] = f"Поставили лайк для записи ID={record_id}"
                        st.rerun()

                    if col2.button("👎", key=f"thumbs_down_{i}"):
                        update_chat_history_rating_by_id(record_id, "-")
                        st.session_state["last_rating_action"] = f"Поставили дизлайк для записи ID={record_id}"
                        st.rerun()
        
    # После ререндера покажем результат последнего действия
    if "last_rating_action" in st.session_state:
        st.info(st.session_state["last_rating_action"])