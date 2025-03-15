# Импорты стандартной библиотеки
import logging
import json
import asyncio
import pandas as pd
import streamlit as st
import time
import random

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
from digital_assistant_first.offergen.utils import get_system_prompt_for_offers, get_system_prompt_for_offers_async
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
    return ChatOpenAI(model=config["Model"], stream=False)

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
        - офферы (если запрос о скидках, промокодах, специальных предложениях, акциях, бонусах, кэшбэке и т.п.)
        - другое (если запрос не подходит ни под одну из перечисленных категорий)
        
        Запрос пользователя: {user_input}
        """
        
        messages = [
            {"role": "system", "content": category_prompt.format(user_input=user_input)}
        ]
        
        # Явно указываем stream=False
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
    offers_data = []
    
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
        
        # Если категория "офферы", запускаем обработку предложений
        if request_category == "офферы":
            try:
                # Прямой асинхронный вызов вместо run_until_complete
                validation_result = await validation_agent.run(user_input)
                validation_result = validation_result.data
                
                if validation_result.number_of_offers_to_generate < 1:
                    validation_result.number_of_offers_to_generate = 10
                
                # Используем асинхронную версию функции
                offers_system_prompt = await get_system_prompt_for_offers_async(validation_result, user_input)
                
                # Если были найдены офферы
                if offers_system_prompt != "No relevant offers were found for the search request.":
                    # Сохраняем информацию об офферах
                    offers_data = {
                        "system_prompt": offers_system_prompt,
                        "validation_result": validation_result
                    }
            except Exception as e:
                logger.error(f"Error in offers processing: {str(e)}", exc_info=True)
                offers_data = []
        
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
        
        # Добавляем специальные инструкции для категории "рестораны"
        restaurant_format_instructions = ""
        if request_category == "рестораны":
            restaurant_format_instructions = """
            ВАЖНО: При ответе на запрос о ресторанах используй следующий формат для представления информации о каждом ресторане:

            Название: [название ресторана]
            Адрес: [полный адрес]
            Режим работы: [часы работы, если есть данные]
            Тип кухни: [какая кухня представлена]
            Средний чек: [стоимость среднего чека, если есть данные]
            Сайт: [официальный сайт, если есть]
            Сайт на рейтинг: [ссылка на страницу с рейтингом]
            Ссылка на отзывы: [ссылка на отзывы]

            Представляй информацию о каждом ресторане в этом формате, с разделением и ясной структурой.
            """
        
        formatted_prompt = system_prompt_template.format(
            context=message_history,
            internet_res=internet_res,
            yandex_res=yandex_res,
            links=links,
            shopping_res=shopping_res,
            telegram_context=telegram_context,
            aviasales_flight_info=aviasales_flight_info,
        )
        
        # Добавляем информацию о категории и инструкции по форматированию в начало промпта
        formatted_prompt = f"{category_info}\n\n{restaurant_format_instructions}\n\n{formatted_prompt}"
        
        # Получаем ответ от модели
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", formatted_prompt),
                ("human", "User query: {input}\nAdditional context: {context}"),
            ]
        )
        messages = prompt_template.format(input=user_input, context="")
        
        # Получаем неасинхронную версию с явно отключенным streaming
        # чтобы обеспечить полноценный ответ для дальнейшей обработки
        response = model.invoke(messages, stream=False)
        
        if hasattr(response, "content"):
            answer = response.content
        elif hasattr(response, "message"):
            answer = response.message.content
        else:
            answer = str(response)
        
        # Проверка и коррекция ссылок
        if config.get("link_checker", False):
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
            "offers_data": offers_data
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

async def handle_user_input(model, config, prompt):
    """Обработать пользовательский ввод и сгенерировать ответ ассистента."""
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = await model_response_generator(model, config)
            
            # Подготавливаем весь контент, который будем стримить
            full_content = []
            
            # Основной ответ
            full_content.append(response["answer"])
            
            # Авиасейлс ссылка
            if "aviasales_link" in response and response["aviasales_link"] and response["aviasales_link"].strip():
                full_content.append(f"\n\n### Данные из Авиасейлс \n **Ссылка** - {response['aviasales_link']}")
            
            # Если категория запроса - рестораны или ивенты, получаем данные для 2GIS
            if response.get("request_category") in ["рестораны", "ивенты"]:
                # Создаем новый синхронный event loop для 2GIS запроса
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Запускаем 2GIS запрос синхронно
                    table_data, pydeck_data = loop.run_until_complete(fetch_2gis_data(prompt, config))
                    
                    # Сохраняем данные для карты сразу для использования позже
                    if pydeck_data and len(pydeck_data) > 0:
                        st.session_state["last_pydeck_data"] = pydeck_data
                        st.session_state["show_map"] = True
                    else:
                        st.session_state["last_pydeck_data"] = []
                        st.session_state["show_map"] = False
                        st.warning("Не найдено точек для отображения на карте.")
                    
                    # ПОДГОТОВКА ТЕКСТОВОЙ ИНФОРМАЦИИ О МЕСТАХ
                    if table_data:
                        places_text = "\n\n📍 Данные о найденных местах 2GIS\n\n"
                        places_text += f"Найдено мест: {len(table_data)}\n\n"
                        
                        # Формируем текстовое описание каждого места
                        for i, place in enumerate(table_data):
                            # Определяем основные данные
                            name = None
                            for name_key in ['Название', 'название', 'name', 'title', 'name_ru', 'Name']:
                                if name_key in place and place[name_key]:
                                    name = place[name_key]
                                    break
                            
                            address = None
                            for addr_key in ['Адрес', 'адрес', 'address', 'address_name', 'full_address', 'Address']:
                                if addr_key in place and place[addr_key]:
                                    address = place[addr_key]
                                    break
                            
                            # Определяем дополнительные данные
                            rating = None
                            if 'Рейтинг' in place and place['Рейтинг']:
                                rating = place['Рейтинг']
                            elif 'rating' in place and place['rating']:
                                rating = place['rating']
                            
                            reviews = None
                            if 'Кол-во Отзывов' in place and place['Кол-во Отзывов']:
                                reviews = place['Кол-во Отзывов']
                            elif 'reviews' in place and place['reviews']:
                                reviews = place['reviews']
                            
                            phone = None
                            if 'phone' in place and place['phone']:
                                phone = place['phone']
                            elif 'Телефон' in place and place['Телефон']:
                                phone = place['Телефон']
                            
                            # Строим точно такое форматирование, как в примере пользователя
                            place_text = f"{i+1}. {name or 'Без названия'} Адрес: {address or 'Не указан'}"
                            
                            if rating:
                                place_text += f" Рейтинг: {rating}"
                            
                            if reviews:
                                place_text += f" | Отзывов: {reviews}"
                                
                            if phone:
                                place_text += f" | Телефон: {phone}"
                            
                            place_text += "\n"
                            places_text += place_text
                    else:
                        places_text += "\n\n*Ничего не найдено в 2GIS.*\n"
                    
                    # ОТОБРАЖАЕМ ТОЛЬКО ТАБЛИЦУ БЕЗ ДУБЛИРОВАНИЯ ТЕКСТОВОЙ ИНФОРМАЦИИ
                    if table_data:
                        # Отображаем только таблицу, текстовую информацию не дублируем
                        # так как она будет отображена через places_text
                        st.subheader("📊 Таблица с полными данными")
                        df = pd.DataFrame(table_data)
                        st.dataframe(df)
                        st.markdown("---")
                
                finally:
                    loop.close()
            
            # Обрабатываем офферы, если они есть
            if "offers_data" in response and response["offers_data"]:
                offers_data = response["offers_data"]
                st.subheader("Генерация офферов")
                
                try:
                    # Используем сохраненный system_prompt для генерации офферов
                    offers_system_prompt = offers_data.get("system_prompt", "")
                    if offers_system_prompt:
                        offers_messages = [
                            {"role": "system", "content": offers_system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        
                        # Получаем ответ
                        offers_response = model.invoke(offers_messages, stream=False)
                        if hasattr(offers_response, "content") and offers_response.content:
                            offers_text = offers_response.content
                        elif hasattr(offers_response, "message") and offers_response.message.content:
                            offers_text = offers_response.message.content
                        else:
                            offers_text = str(offers_response)
                        
                        # Отображаем офферы
                        st.markdown(offers_text)
                    else:
                        st.warning("Не удалось сгенерировать офферы для вашего запроса.")
                except Exception as e:
                    logger.error(f"Error generating offers: {str(e)}", exc_info=True)
                    st.error("Произошла ошибка при генерации офферов.")
            
            # КАРТА - выводим В САМОМ КОНЦЕ функции, после всего остального
            if response.get("request_category") in ["рестораны", "ивенты"] and st.session_state.get("show_map", False):
                if st.session_state.get("last_pydeck_data", []):
                    pydeck_data = st.session_state["last_pydeck_data"]
                    if len(pydeck_data) > 0:
                        # Создаем отдельный контейнер для карты
                        with st.container():
                            st.markdown("## ")
                            st.subheader("🗺️ Интерактивная карта")
                            # Добавляем разделитель перед картой
                            st.markdown("---")
                            
                            df_pydeck = pd.DataFrame(pydeck_data)
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

            st.session_state["messages"].append(
                {
                    "role": "assistant", 
                    "content": response_text, 
                    "question": prompt,
                    "show_map": st.session_state.get("show_map", False),
                    "request_category": response.get("request_category", ""),
                    "pydeck_data": response.get("pydeck_data", []),  # Store pydeck data with the message
                    "record_id": None  # Will be set after DB insert
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

            # Основной контент сообщения - для ассистента выводим полный контент
            if message["role"] == "assistant":
                # Для всех сообщений ассистента отображаем полный контент
                # без разделения на части, так как дублирование уже устранено
                st.markdown(message["content"])
                
                # УДАЛЯЕМ ОТОБРАЖЕНИЕ ТАБЛИЦЫ
                # Оставляем только данные в объекте message для других компонентов
            else:
                # Для сообщений пользователя просто отображаем контент
                st.markdown(message["content"])
            
            # Если это ассистент, обрабатываем рейтинги и карты
            if message["role"] == "assistant":
                # Показываем карту для всех сообщений о ресторанах и ивентах
                is_map_needed = message.get("request_category") in ["рестораны", "ивенты"] or message.get("show_map", False)
                
                if is_map_needed:
                    # Получаем данные карты из сообщения или из session_state
                    if i == last_assistant_index and "last_pydeck_data" in st.session_state:
                        pydeck_data = st.session_state["last_pydeck_data"]
                        if "pydeck_data" not in message:
                            message["pydeck_data"] = pydeck_data
                    elif "pydeck_data" in message:
                        pydeck_data = message["pydeck_data"]
                    else:
                        pydeck_data = st.session_state.get("last_pydeck_data", [])
                        
                    if pydeck_data and len(pydeck_data) > 0:
                        with st.container():
                            st.markdown("## ")
                            st.subheader("🗺️ Интерактивная карта")
                            st.markdown("---")
                            
                            df_pydeck = pd.DataFrame(pydeck_data)
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

def model_response_generator_sync(model, config):
    """Сгенерировать ответ с использованием модели и ретривера синхронно."""
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

    # Определение категории запроса с помощью агента - синхронная версия
    def categorize_request():
        category_prompt = """
        Определи категорию запроса пользователя и верни ТОЛЬКО одну из следующих категорий без дополнительных пояснений:
        - рестораны (если запрос о ресторанах, кафе, еде, доставке питания и т.п.)
        - ивенты (если запрос о мероприятиях, концертах, выставках, фестивалях и т.п.)
        - поездки (если запрос о поездках на машинах, такси, аренде автомобилей и т.п.)
        - офферы (если запрос о скидках, промокодах, специальных предложениях, акциях, бонусах, кэшбэке и т.п.)
        - другое (если запрос не подходит ни под одну из перечисленных категорий)
        
        Запрос пользователя: {user_input}
        """
        
        messages = [
            {"role": "system", "content": category_prompt.format(user_input=user_input)}
        ]
        
        # Явно указываем stream=False
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
    
    # Получаем категорию запроса синхронно
    request_category = categorize_request()
    
    # Инициализируем переменные по умолчанию
    shopping_res = ""
    internet_res = ""
    links = ""
    yandex_res = ""
    telegram_context = ""
    table_data = []
    pydeck_data = []
    offers_data = []
    aviasales_url = ""
    aviasales_flight_info = ""
    
    # Создаем loop для асинхронных вызовов внутри синхронной функции
    # В одном месте вместо распределенных вызовов
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Для category = поездки или офферы получим необходимые данные
        if request_category == "поездки" or request_category == "другое":
            aviasales_tool = AviasalesHandler()
            tickets_need = loop.run_until_complete(aviasales_tool.aviasales_request(model, config, user_input))
            
            if tickets_need.get("response", "").lower() == "true":
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
                    aviasales_flight_info = loop.run_until_complete(
                        aviasales_tool.get_info_aviasales_url(aviasales_url=aviasales_url, user_input=user_input)
                    )
        
        # Для офферов
        if request_category == "офферы":
            try:
                validation_result = loop.run_until_complete(validation_agent.run(user_input))
                validation_result = validation_result.data
                
                if validation_result.number_of_offers_to_generate < 1:
                    validation_result.number_of_offers_to_generate = 10
                
                offers_system_prompt = loop.run_until_complete(
                    get_system_prompt_for_offers_async(validation_result, user_input)
                )
                
                if offers_system_prompt != "No relevant offers were found for the search request.":
                    offers_data = {
                        "system_prompt": offers_system_prompt,
                        "validation_result": validation_result
                    }
            except Exception as e:
                logger.error(f"Error in offers processing: {str(e)}", exc_info=True)
                offers_data = []
    
    finally:
        loop.close()
        
    # Формируем системный промпт
    system_prompt_template = config["system_prompt"]
    
    # Создаем информацию о категории запроса
    category_info = f"Категория запроса пользователя: {request_category}"
    
    # Добавляем специальные инструкции для категории "рестораны"
    restaurant_format_instructions = ""
    if request_category == "рестораны":
        restaurant_format_instructions = """
        ВАЖНО: При ответе на запрос о ресторанах используй следующий формат для представления информации о каждом ресторане:

        Название: [название ресторана]
        Адрес: [полный адрес]
        Режим работы: [часы работы, если есть данные]
        Тип кухни: [какая кухня представлена]
        Средний чек: [стоимость среднего чека, если есть данные]
        Сайт: [официальный сайт, если есть]
        Сайт на рейтинг: [ссылка на страницу с рейтингом]
        Ссылка на отзывы: [ссылка на отзывы]

        Представляй информацию о каждом ресторане в этом формате, с разделением и ясной структурой.
        """
    
    formatted_prompt = system_prompt_template.format(
        context=message_history,
        internet_res=internet_res,
        yandex_res=yandex_res,
        links=links,
        shopping_res=shopping_res,
        telegram_context=telegram_context,
        aviasales_flight_info=aviasales_flight_info,
    )
    
    # Добавляем информацию о категории и инструкции по форматированию в начало промпта
    formatted_prompt = f"{category_info}\n\n{restaurant_format_instructions}\n\n{formatted_prompt}"
    
    # Получаем ответ от модели
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", formatted_prompt),
            ("human", "User query: {input}\nAdditional context: {context}"),
        ]
    )
    messages = prompt_template.format(input=user_input, context="")
    
    # Получаем неасинхронную версию с явно отключенным streaming
    response = model.invoke(messages, stream=False)
    
    if hasattr(response, "content"):
        answer = response.content
    elif hasattr(response, "message"):
        answer = response.message.content
    else:
        answer = str(response)
    
    log_api_call(
        logger=logger,
        source=f"LLM ({config['Model']})",
        request=user_input,
        response=answer,
    )
    
    return {
        "answer": answer,
        "aviasales_link": aviasales_url,
        "table_data": table_data,
        "pydeck_data": pydeck_data,
        "request_category": request_category,
        "offers_data": offers_data
    }

def handle_user_input_sync(model, config, prompt):
    """Обработать пользовательский ввод и сгенерировать ответ ассистента (синхронная версия)."""
    if prompt:
        # Всегда сбрасываем данные карты и таблицы перед новым запросом
        st.session_state["last_pydeck_data"] = []
        st.session_state["show_map"] = False
        st.session_state["last_2gis_query"] = prompt  # Сохраняем текущий запрос
        
        # Получаем предварительную категорию запроса
        categorize_prompt = """
        Определи категорию запроса пользователя и верни ТОЛЬКО одну из следующих категорий без дополнительных пояснений:
        - рестораны (если запрос о ресторанах, кафе, еде, доставке питания и т.п.)
        - ивенты (если запрос о мероприятиях, концертах, выставках, фестивалях и т.п.)
        - другое (если запрос не подходит ни под одну из перечисленных категорий)
        
        Запрос пользователя: {prompt}
        """
        
        messages = [
            {"role": "system", "content": categorize_prompt.format(prompt=prompt)}
        ]
        
        # Получаем предварительную категорию
        pre_category = model.invoke(messages, stream=False).content.strip().lower()
        
        # Сбрасываем флаг show_map и данные карты если запрос НЕ о ресторанах/ивентах
        if pre_category != "рестораны" and pre_category != "ивенты":
            st.session_state["show_map"] = False
            st.session_state["last_pydeck_data"] = []
            
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Используем синхронную версию генератора ответов
            response = model_response_generator_sync(model, config)
            
            # Основной ответ для отображения в интерфейсе
            answer_text = response["answer"]
            aviasales_text = ""
            places_text = ""
            offers_text = ""
            
            # Отображаем данные Aviasales, если они есть
            if "aviasales_link" in response and response["aviasales_link"] and response["aviasales_link"].strip():
                aviasales_text = f"\n\n### Данные из Авиасейлс \n **Ссылка** - {response['aviasales_link']}"
            
            # Если категория запроса - рестораны или ивенты, получаем данные для 2GIS
            table_data = []
            if response.get("request_category") in ["рестораны", "ивенты"]:
                # Создаем новый синхронный event loop для 2GIS запроса
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Добавляем аннотацию @st.cache_data(ttl=60) выше функции fetch_2gis_data в другом файле
                    # или форсируем новый запрос здесь:
                    st.session_state["2gis_cache_key"] = f"{prompt}_{time.time()}"  # Уникальный ключ
                    table_data, pydeck_data = loop.run_until_complete(fetch_2gis_data(prompt, config))
                    
                    # Проверяем и сохраняем новые данные
                    if pydeck_data and len(pydeck_data) > 0:
                        # Печатаем отладочную информацию
                        print(f"Новые данные карты: {len(pydeck_data)} точек")
                        print(f"Координаты первой точки: lat={pydeck_data[0]['lat']}, lon={pydeck_data[0]['lon']}")
                        
                        st.session_state["last_pydeck_data"] = pydeck_data.copy()  # Создаем копию
                        st.session_state["show_map"] = True
                    else:
                        st.session_state["last_pydeck_data"] = []
                        st.session_state["show_map"] = False
                        places_text += "\n\n*Не найдено точек для отображения на карте.*"
                    
                    # ПОДГОТОВКА ТЕКСТОВОЙ ИНФОРМАЦИИ О МЕСТАХ
                    if table_data:
                        places_text += "\n\n**📍 Данные о найденных местах 2GIS API**\n\n"
                        places_text += f"Найдено мест: {len(table_data)}\n\n"
                        
                        # Формируем текстовое описание каждого места
                        for i, place in enumerate(table_data):
                            # Определяем основные данные
                            name = None
                            for name_key in ['Название', 'название', 'name', 'title', 'name_ru', 'Name']:
                                if name_key in place and place[name_key]:
                                    name = place[name_key]
                                    break
                            
                            address = None
                            for addr_key in ['Адрес', 'адрес', 'address', 'address_name', 'full_address', 'Address']:
                                if addr_key in place and place[addr_key]:
                                    address = place[addr_key]
                                    break
                            
                            # Определяем дополнительные данные
                            rating = None
                            if 'Рейтинг' in place and place['Рейтинг']:
                                rating = place['Рейтинг']
                            elif 'rating' in place and place['rating']:
                                rating = place['rating']
                            
                            reviews = None
                            if 'Кол-во Отзывов' in place and place['Кол-во Отзывов']:
                                reviews = place['Кол-во Отзывов']
                            elif 'reviews' in place and place['reviews']:
                                reviews = place['reviews']
                            
                            phone = None
                            if 'phone' in place and place['phone']:
                                phone = place['phone']
                            elif 'Телефон' in place and place['Телефон']:
                                phone = place['Телефон']
                            
                            # Строим форматированное описание с жирным названием
                            place_text = f"{i+1}. **{name or 'Без названия'}** Адрес: {address or 'Не указан'}"
                            
                            if rating:
                                place_text += f" Рейтинг: {rating}"
                            
                            if reviews:
                                place_text += f" | Отзывов: {reviews}"
                                
                            if phone:
                                place_text += f" | Телефон: {phone}"
                            
                            place_text += "\n"
                            places_text += place_text
                    else:
                        places_text += "\n\n*Ничего не найдено в 2GIS.*\n"
                
                finally:
                    loop.close()
            
            # Обрабатываем офферы, если они есть
            if "offers_data" in response and response["offers_data"]:
                offers_data = response["offers_data"]
                offers_text += "\n\n## Генерация офферов\n"
                
                try:
                    offers_system_prompt = offers_data.get("system_prompt", "")
                    if offers_system_prompt:
                        offers_messages = [
                            {"role": "system", "content": offers_system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        
                        offers_response = model.invoke(offers_messages, stream=False)
                        if hasattr(offers_response, "content") and offers_response.content:
                            generated_offers = offers_response.content
                        elif hasattr(offers_response, "message") and offers_response.message.content:
                            generated_offers = offers_response.message.content
                        else:
                            generated_offers = str(offers_response)
                        
                        offers_text += generated_offers
                    else:
                        offers_text += "\n\n*Не удалось сгенерировать офферы для вашего запроса.*"
                except Exception as e:
                    logger.error(f"Error generating offers: {str(e)}", exc_info=True)
                    offers_text += "\n\n*Произошла ошибка при генерации офферов.*"
            
            # Собираем полный ответ для стриминга - основной ответ + места + авиасейлс
            full_response_text = answer_text + places_text + aviasales_text + offers_text
            
            # Создаем плейсхолдер для потокового текста
            text_placeholder = st.empty()
            
            # Имитируем печатную машинку с помощью плейсхолдера
            
            # Разбиваем текст на части для имитации печати
            display_text = ""
            for i, char in enumerate(full_response_text):
                display_text += char
                
                # Обновляем текст с разной частотой для лучшего эффекта
                if i % 2 == 0 or char in ['.', '!', '?', '\n']:
                    text_placeholder.markdown(display_text)
                    
                    # Задержка между символами (варьируется)
                    delay = 0.01  # Базовая задержка
                    
                    # Более длинная пауза после знаков препинания
                    if char in ['.', '!', '?']:
                        delay = 0.05
                    elif char == '\n':
                        delay = 0.03
                    
                    # Добавляем небольшую случайность
                    time.sleep(delay * random.uniform(0.5, 1.5))
            
            # Устанавливаем финальный текст для сохранения
            response_text = full_response_text
            
            # КАРТА - выводим В САМОМ КОНЦЕ функции, после всего остального
            if response.get("request_category") in ["рестораны", "ивенты"]:
                if st.session_state.get("last_pydeck_data", []) and len(st.session_state["last_pydeck_data"]) > 0:
                    pydeck_data = st.session_state["last_pydeck_data"]
                    if len(pydeck_data) > 0:
                        with st.container():
                            st.markdown("## ")
                            st.subheader("🗺️ Интерактивная карта 2GIS")
                            st.markdown("---")
                            
                            df_pydeck = pd.DataFrame(pydeck_data)
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

            # Отображаем офферы в интерфейсе - оставляем офферы с отдельным выводом
            if "offers_data" in response and response["offers_data"] and offers_text:
                st.subheader("Генерация офферов")
                offers_data = response["offers_data"]
                try:
                    offers_system_prompt = offers_data.get("system_prompt", "")
                    if offers_system_prompt:
                        # Офферы уже включены в основной ответ, не показываем дублированно
                        pass
                    else:
                        st.warning("Не удалось сгенерировать офферы для вашего запроса.")
                except Exception as e:
                    st.error("Произошла ошибка при генерации офферов.")
            
            # Сохраняем дополнительную информацию для истории сообщений
            st.session_state["messages"].append(
                {
                    "role": "assistant", 
                    "content": response_text,  # Сохраняем полный текст, включая информацию о местах
                    "question": prompt,
                    "request_category": response.get("request_category", ""),
                    "show_map": st.session_state.get("show_map", False),
                    "pydeck_data": st.session_state.get("last_pydeck_data", []),
                    "places_text": places_text,  # Сохраняем текстовую информацию о местах отдельно
                    "aviasales_text": aviasales_text,
                    "offers_text": offers_text,
                    "table_data": table_data if 'table_data' in locals() else [],  # Сохраняем данные таблицы
                    "record_id": None  # Will be set after DB insert
                }
            )
            
            # Добавляем оценку ответа
            col1, col2 = st.columns(2)
            if col1.button("👍", key=f"thumbs_up_{len(st.session_state['messages'])}"):
                st.success("Вы поставили 👍")
            if col2.button("👎", key=f"thumbs_down_{len(st.session_state['messages'])}"):
                st.error("Вы поставили 👎")  

            # Сохраняем в базу данных
            record_id = insert_chat_history_return_id(
                user_query=prompt,
                model_response=response_text,  # Сохраняем полный текст, включая информацию о местах
                mode=config["mode"],
                rating=None
            )

            # Обновляем record_id в сообщении
            st.session_state["messages"][-1]["record_id"] = record_id