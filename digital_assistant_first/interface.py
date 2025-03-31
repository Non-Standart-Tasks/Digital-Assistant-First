# Импорты стандартной библиотеки
import logging
import json
import asyncio
import pandas as pd
import streamlit as st
import time
import random
import requests
from openai import OpenAI  # Добавляем прямой импорт OpenAI
from digital_assistant_first.multiagent_system.deepsearch import deepsearch
from agents import Agent, function_tool, Runner, set_tracing_disabled, OpenAIChatCompletionsModel, WebSearchTool, RunContextWrapper
from pydantic import BaseModel

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
from digital_assistant_first.geo_system.two_gis import fetch_2gis_data, build_route_from_query
from digital_assistant_first.offergen.agent import validation_agent
from digital_assistant_first.offergen.utils import get_offers_data
from digital_assistant_first.yndx_system.restaurant_context import fetch_yndx_context
from digital_assistant_first.utils.link_checker import link_checker, corrector
from digital_assistant_first.utils.database import (
    init_db, 
    insert_chat_history_return_id, 
    update_chat_history_rating_by_id, 
    get_chat_record_by_id
)
from digital_assistant_first.geo_system.map_display import display_2gis_map
from openai import AsyncOpenAI
from dotenv import load_dotenv

class Category(BaseModel):
    category: str

load_dotenv()

logger = setup_logging(logging_path="logs/digital_assistant.log")
#serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.csv")
init_db()

client = AsyncOpenAI(api_key="sk-13dd9563da2d435ebd38c2693dd78c6f", base_url="https://api.deepseek.com")


# Add the initialize_model function here to avoid circular import
def initialize_model(config):
    """Инициализация языковой модели на основе конфигурации."""
    # Инициализируем LangChain модель для совместимости с существующим кодом
    langchain_model = ChatOpenAI(model=config["Model"], stream=False)
    
    # Инициализируем прямой клиент OpenAI для возможности использования web_search_preview
    openai_client = OpenAI()
    
    # Сохраняем клиент в конфигурации
    config["openai_client"] = openai_client
    
    return langchain_model

def init_message_history(template_prompt):
    """Инициализировать историю сообщений для чата."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Always show the system message regardless of session state
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
            if "question" in message:
                st.markdown(f"**Вопрос**: {message['question']}")

            if message["role"] == "assistant":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])
            
            if message["role"] == "assistant":
                is_map_needed = message.get("request_category") in ["рестораны", "ивенты", "маршруты"] or message.get("show_map", False)
                
                if is_map_needed:
                    map_type = message.get("map_type", "points")
                    if map_type is None:
                        map_type = "points"
                    
                    if map_type == "points":
                        if i == last_assistant_index and "last_pydeck_data" in st.session_state:
                            pydeck_data = st.session_state["last_pydeck_data"]
                            if "pydeck_data" not in message:
                                message["pydeck_data"] = pydeck_data
                        elif "pydeck_data" in message:
                            pydeck_data = message["pydeck_data"]
                        else:
                            pydeck_data = st.session_state.get("last_pydeck_data", [])
                            
                        if pydeck_data and len(pydeck_data) > 0:
                            display_2gis_map(
                                pydeck_data=pydeck_data,
                                map_type="points",
                                title="🗺️ Интерактивная карта"
                            )
                    
                    elif map_type == "route":
                        path_points = message.get("path_points", [])
                        route_points = message.get("route_points", [])
                        
                        if i == last_assistant_index:
                            if not path_points and "path_points" in st.session_state:
                                path_points = st.session_state["path_points"]
                                message["path_points"] = path_points
                            
                            if not route_points and "route_points" in st.session_state:
                                route_points = st.session_state["route_points"]
                                message["route_points"] = route_points
                        
                        if path_points and route_points and len(path_points) > 0 and len(route_points) > 0:
                            display_2gis_map(
                                pydeck_data=[],  # Empty for route type
                                map_type="route",
                                path_points=path_points,
                                route_points=route_points,
                                title="🗺️ Построенный маршрут"
                            )
                
                record_id = message.get("record_id")
                if record_id:
                    col1, col2, col3 = st.columns(3)

                    if col1.button("👍", key=f"thumbs_up_{i}"):
                        update_chat_history_rating_by_id(record_id, "+")
                        st.session_state["last_rating_action"] = f"Поставили лайк для записи ID={record_id}"
                        st.rerun()

                    if col2.button("👎", key=f"thumbs_down_{i}"):
                        update_chat_history_rating_by_id(record_id, "-")
                        st.session_state["last_rating_action"] = f"Поставили дизлайк для записи ID={record_id}"
                        st.rerun()
                    # Добавляем кнопку генерации оффера
                    col3.link_button(
                        "🎁 Сгенерировать оффер", 
                        st.session_state["messages"][i].get("offers_link", "https://google.com"), 
                        use_container_width=True
                    )
        
    if "last_rating_action" in st.session_state:
        st.info(st.session_state["last_rating_action"])

def model_response_generator_sync(model, config, status_placeholder):
    status_placeholder.info("🤔 Анализирую ваш запрос...")
    time.sleep(1)
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
    
    # Получаем категорию запроса синхронно
    agent_address = Agent(
    name="Assistant",
    instructions=""" 
    Определи категорию запроса пользователя и верни ТОЛЬКО одну из следующих категорий без дополнительных пояснений:
        - рестораны (если запрос о ресторанах, кафе, еде в общественных местах)
        - бары (если запрос о барах, пабах, винных барах)
        - кальянные (если запрос о кальянных, кальян-барах)
        - доставка_еды (если запрос о доставке еды, заказе еды на дом)
        - банкет (если запрос о проведении банкета, юбилея, корпоратива, аренде зала для торжества)
        - кейтеринг (если запрос о выездном обслуживании, доставке готовых блюд на мероприятие)
        - ивенты (если запрос о мероприятиях, концертах, выставках, фестивалях)
        - маршруты (если запрос о том, как построить маршрут, проложить путь между местами)
        - поездки (если запрос о поездках на машинах, такси, аренде автомобилей, авиабилетах, железнодорожных билетах)
        - другое (если запрос не подходит ни под одну из перечисленных категорий)
    """,
    model="gpt-4o-mini",
    output_type=Category)
    
    # Инициализируем переменные по умолчанию
    # Пока эти штуки оставим.
    shopping_res = ""
    internet_res = ""
    links = ""
    yandex_res = ""
    telegram_context = ""
    table_data = []
    pydeck_data = []
    offers_data = {}  # Инициализируем как пустой словарь вместо пустого списка
    aviasales_url = ""
    aviasales_flight_info = ""

    global_prompt = config.get("global_prompt", "").format(context=message_history)
    
    # Создаем loop для асинхронных вызовов внутри синхронной функции
    # В одном месте вместо распределенных вызовов
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
           
    try:
        request_category = Runner.run_sync(agent_address, user_input)
        request_category = request_category.final_output.category

        category_emoji = {
            "рестораны": "🍽️",
            "бары": "🍺",
            "кальянные": "💨",
            "доставка_еды": "🛵",
            "банкет": "🎉",
            "кейтеринг": "🍱",
            "ивенты": "🎭",
            "маршруты": "🗺️",
            "поездки": "✈️",
            "другое": "📋"
        }

        emoji = category_emoji.get(request_category, "📋")
        status_placeholder.info(f"{emoji} Определена категория: {request_category}")
        
        #Пока выключим интернет поиск
        #if config.get("internet_search", False):
        #    async def fetch_internet_data():
        #        _, serpapi_key = serpapi_key_manager.get_best_api_key()
                
                # Добавляем информацию о категории к запросу для более точного поиска
                #enhanced_query = user_input
                #if request_category != "другое":
                #    enhanced_query = f"{user_input} {request_category}"
                    
                #shopping = await search_shopping(enhanced_query, serpapi_key)
                #internet, links_data, _ = await search_places(enhanced_query, serpapi_key)
                #yandex_res = await yandex_search(enhanced_query, serpapi_key)
                #return shopping, internet, links_data, yandex_res
            
            # Запускаем асинхронную функцию через event loop
            #shopping_res, internet_res, links, yandex_res = loop.run_until_complete(fetch_internet_data())
        #else:
            #   shopping_res, internet_res, links, yandex_res = "", "", [], []
            
            
        # Для category = поездки или офферы получим необходимые данные
        # aviasales_flight_info мы будем заменять!!!
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
        
        # Для офферов - при включенном toggle обрабатываем независимо от категории запроса
        # Код для офферов - тут гоняем РАГ
        if config.get("offers_enabled", False):
            try:
                validation_result = loop.run_until_complete(validation_agent.run(user_input))
                validation_result = validation_result.data
                
                if validation_result.number_of_offers_to_generate < 1:
                    validation_result.number_of_offers_to_generate = 10
                
                offers_data, offer_json = loop.run_until_complete(
                    get_offers_data(validation_result, user_input)
                )
                
                agent_offers = Agent(
                    name="Assistant",
                    instructions="""
                    Ты - форматировщик офферов VTB Family. Твоя задача:
                    1. Форматировать офферы в markdown
                    2. Структурировать каждый оффер четко и ясно
                    3. Писать все на русском языке
                    4. Выводить только релевантные офферы
                    
                    ВАЖНО: Не добавляй никаких итогов, резюме или вступлений. 
                    Выводи ТОЛЬКО форматированные офферы.
                    
                    Используй формат:
                    ### [Название оффера]
                    **Категория:** [Категория]
                    
                    **Описание предложения:**
                    [Краткое описание основной скидки/предложения]
                    
                    **Информация о компании:**
                    - Адрес: [Адрес если есть]
                    - Телефон: [Телефон если есть]
                    - Сайт: [Сайт если есть]
                    
                    **Ссылка на предложение:** [Подробнее на VTB Family](URL)
                    
                    ---
                    """,
                    model=config["Model"]
                )

                offers_response = loop.run_until_complete(Runner.run(agent_offers, offers_data))
                offers_text = offers_response.final_output
                if offers_data != "No relevant offers were found for the search request.":
                    try:
                        response = requests.post("http://localhost:8001/generate_link", json=offer_json)
                        response.raise_for_status()
                        link = response.json().get("link")
                        logger.info(f"Ссылка на Streamlit-приложение: {link}")
                    except Exception as e:
                        logger.error(f"Ошибка при отправке данных в микросервис: {e}")
                    #logger.info(f'проверка перед отправкой в JSON{offer_json}')
                    offers_data = {
                        "offers_text": offers_text,
                        "validation_result": validation_result,
                        "offers_link":link
                    }
            except Exception as e:
                logger.error(f"Error in offers processing: {str(e)}", exc_info=True)
                offers_data = {}  # Инициализируем как пустой словарь вместо пустого списка

        if config.get("deepsearch", False):
            deepsearch_res = loop.run_until_complete(deepsearch(user_input, status_placeholder, config))

        else:
            web_search_context_size = config.get("web_search_context_size", "medium")
        
            # Используем нативный веб-поиск OpenAI
            logger.info(f"Используем нативный веб-поиск OpenAI для запроса: {user_input}")
            
            agent_web_seach = Agent(
                name="Assistant",
                instructions=f"""
                Ответь на вопрос пользователя, в зависимости от категории запроса: используя интернет и контекст. 
                
                Если запрос связан с выводом каких-либо мест, то выведи столько вариантов, сколько попросил пользовательл
                если явно количество не указано, то выведи 5 вариантов.

                ОБЯЗАТЕЛЬНО СТАРАЙСЯ ВЫВОДИТЬ ССЫЛКИ И ТОЛЬКО РАБОЧИЕ ССЫЛКИ.

                """,
                model='gpt-4o-mini',
                tools=[WebSearchTool(search_context_size=web_search_context_size)])
            
            web_search_response = Runner.run_sync(agent_web_seach, user_input)
            web_search_response = web_search_response.final_output
            
            print(f"DEBUG: Ответ от веб-поиска: {web_search_response}")
            
            log_api_call(
                logger=logger,
                source=f"LLM ({config['Model']})",
                request=user_input,
                response=web_search_response,
            )
            
            
    finally:
        loop.close()
        
    if config.get("deepsearch", False):
        return {
        "answer": deepsearch_res,
        "aviasales_link": aviasales_url,
        "table_data": table_data,
        "pydeck_data": pydeck_data,
        "request_category": request_category,
        "offers_data": offers_data
    }

    else:
        return {
        "answer": web_search_response,
        "aviasales_link": aviasales_url,
        "table_data": table_data,
        "pydeck_data": pydeck_data,
        "request_category": request_category,
        "offers_data": offers_data
    }

def handle_user_input_sync(model, config, prompt):
    """Обработать пользовательский ввод и сгенерировать ответ ассистента (синхронная версия)."""
    if prompt:
        status_placeholder = st.empty()
        # Всегда сбрасываем данные карты и таблицы перед новым запросом
        st.session_state["last_pydeck_data"] = []
        st.session_state["show_map"] = False
        st.session_state["last_2gis_query"] = prompt  # Сохраняем текущий запрос
        

        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Используем синхронную версию генератора ответов
            response = model_response_generator_sync(model, config, status_placeholder)
            
            # Основной ответ для отображения в интерфейсе
            answer_text = response["answer"]
            aviasales_text = ""
            places_text = ""
            offers_text = ""
            
            pre_category = response["request_category"]
            
            #if pre_category not in ["рестораны", "ивенты", "маршруты"]:
                #print(f"DEBUG start: Сбрасываем данные карты - запрос не о ресторанах/ивентах/маршрутах")
                #st.session_state["show_map"] = False
                #st.session_state["last_pydeck_data"] = []
            
            # Для маршрутов очищаем предыдущие данные, но сохраняем флаг типа
            if pre_category == "маршруты":
                print(f"DEBUG start: Предварительно определен запрос о маршрутах")
                # Очищаем старые данные
                if "path_points" in st.session_state:
                    st.session_state.pop("path_points")
                if "route_points" in st.session_state:
                    st.session_state.pop("route_points")
                if "route_info" in st.session_state:
                    st.session_state.pop("route_info")
                # Устанавливаем тип карты, но не показываем до получения данных
                st.session_state["map_type"] = "route"
                st.session_state["show_map"] = False
                
            # Отображаем данные Aviasales, если они есть
            if "aviasales_link" in response and response["aviasales_link"] and response["aviasales_link"].strip():
                aviasales_text = f"\n\n### Данные из Авиасейлс \n **Ссылка** - {response['aviasales_link']}"
            
            # Если категория запроса - рестораны или ивенты И включен поиск по 2GIS, получаем данные 2GIS
            table_data = []
            pydeck_data = []
            path_points = []
            route_info = None
            
            # Проверяем, включен ли поиск по 2GIS
            maps_2gis_enabled = config.get("maps_2gis_enabled", False)
            
            if maps_2gis_enabled:   
                # Создаем новый синхронный event loop для 2GIS запроса
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Форсируем новый запрос через уникальный ключ
                    st.session_state["2gis_cache_key"] = f"{prompt}_{time.time()}"
                    table_data, pydeck_data = loop.run_until_complete(fetch_2gis_data(prompt, config))
                    
                    # Проверяем и сохраняем новые данные
                    if pydeck_data and len(pydeck_data) > 0:
                        # Печатаем отладочную информацию
                        print(f"Новые данные карты: {len(pydeck_data)} точек")
                        print(f"Координаты первой точки: lat={pydeck_data[0]['lat']}, lon={pydeck_data[0]['lon']}")
                        
                        st.session_state["last_pydeck_data"] = pydeck_data.copy()  # Создаем копию
                        st.session_state["show_map"] = True
                        st.session_state["map_type"] = "points"  # Тип карты - точки
                    else:
                        st.session_state["last_pydeck_data"] = []
                        st.session_state["show_map"] = False
                        st.session_state["map_type"] = None
                        places_text += "\n\n*Не найдено точек для отображения на карте.*"
                    
                    # ПОДГОТОВКА ТЕКСТОВОЙ ИНФОРМАЦИИ О МЕСТАХ
                    if table_data:
                        places_text += "\n\n### 📍 Данные о найденных местах 2GIS API\n"
                        places_text += f"Найдено мест: {len(table_data)}\n\n"
                        
                        # Формируем текстовое описание каждого места
                        for i, place in enumerate(table_data):
                            # Определяем основные данные
                            name = place.get("name", "Без названия")
                            address = place.get("address", "Адрес не указан")
                            rating = place.get("rating", "Нет данных")
                            reviews = place.get("reviews", "Нет данных")
                            phone = place.get("phone", "")
                            cuisine = place.get("cuisine", "Не указано")
                            schedule = place.get("schedule", "Не указано")
                            
                            # Строим форматированное описание с базовой информацией на одной строке
                            place_text = f"{i+1}. {name} Адрес: {address}"
                            
                            if rating and rating != 0:
                                place_text += f" Рейтинг: {rating}"
                            
                            if reviews and reviews != 0:
                                place_text += f" | Отзывов: {reviews}"
                                
                            # Добавляем дополнительные данные на новых строках
                            if phone:
                                place_text += f"\n   📞 Телефон: {phone}"
                                
                            if cuisine and cuisine != "Не указано":
                                place_text += f"\n   🍽️ Кухня: {cuisine}"
                                
                            if schedule and schedule != "Не указано":
                                place_text += f"\n   🕒 Режим работы: {schedule}"
                            
                            place_text += "\n\n"
                            places_text += place_text
                    else:
                        places_text += "\n\n*Ничего не найдено в 2GIS.*\n"
                
                finally:
                    loop.close()
            
            # Если категория запроса - маршруты И включен поиск по 2GIS, получаем данные для построения маршрута
            elif maps_2gis_enabled and response.get("request_category") == "маршруты":
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Запускаем построение маршрута
                    from digital_assistant_first.geo_system.two_gis import build_route_from_query
                    print(f"DEBUG: Запрос маршрута для: {prompt}")
                    route_info, path_points, points_data, route_details = loop.run_until_complete(build_route_from_query(prompt, config))
                    
                    # Добавляем отладочную информацию
                    print(f"DEBUG: Результаты запроса маршрута:")
                    print(f"DEBUG: route_info: {route_info}")
                    print(f"DEBUG: points count: {len(path_points) if path_points else 0}")
                    
                    # Если маршрут построен успешно
                    if route_info and path_points and len(path_points) > 0:
                        # Сохраняем данные о маршруте для отображения
                        st.session_state["route_info"] = route_info
                        st.session_state["path_points"] = path_points
                        st.session_state["route_points"] = points_data
                        st.session_state["route_details"] = route_details
                        st.session_state["map_type"] = "route"
                        st.session_state["show_map"] = True
                        
                        # Добавляем информацию о маршруте в текст ответа
                        route_text = f"\n\n🚗 **Маршрут построен!**\n" \
                                    f"Расстояние: {route_info['distance']/1000:.1f} км\n" \
                                    f"Примерное время в пути: {route_info['duration']//60} мин\n"
                        
                        # Добавляем навигационные инструкции, если они есть
                        if route_details and "instructions_text" in route_details and route_details["instructions_text"]:
                            route_text += "\n**Навигационные инструкции:**\n" + "\n".join(route_details["instructions_text"])
                        
                        answer_text += route_text
                finally:
                    loop.close()
            else:
                # Если 2GIS отключен или категория не подходящая, сбрасываем флаги отображения карты
                st.session_state["show_map"] = False
                st.session_state["map_type"] = None
                st.session_state["last_pydeck_data"] = []

            # Собираем полный ответ для стриминга - основной ответ + места + авиасейлс
            full_response_text = answer_text + places_text + aviasales_text
            
            # Создаем плейсхолдер для потокового текста
            text_placeholder = st.empty()
            display_text = ""  # Move this outside the function
            
            def stream_text(text_to_stream, current_display_text=""):
                display_text = current_display_text
                for i, char in enumerate(text_to_stream):
                    display_text += char
                    
                    if i % 2 == 0 or char in ['.', '!', '?', '\n']:
                        text_placeholder.markdown(display_text)
                        
                        delay = 0.01
                        if char in ['.', '!', '?']:
                            delay = 0.05
                        elif char == '\n':
                            delay = 0.03
                        
                        time.sleep(delay * random.uniform(0.5, 1.5))
                return display_text

            # Stream the initial response
            display_text = stream_text(full_response_text)
            
            # Обрабатываем офферы после основного текста
            if config.get("offers_enabled", False):
                try:
                    offers_text = response["offers_data"]['offers_text']
                    logger.info(f"Offers data: {offers_text}")  # Debug log
                    offers_link = response["offers_data"]['offers_link']
                    offers_section = f"\n\n### 🎁 Специальные предложения VTB Family\n{offers_text}"
                    display_text = stream_text(offers_section, display_text)  # Pass current display_text
                            
                    # Обновляем полный текст ответа
                    response_text = display_text
                    
                except Exception as e:
                    logger.error(f"Error generating offers: {str(e)}", exc_info=True)
                    st.error("Произошла ошибка при генерации офферов.")
                
                else:
                    logger.info("No valid offers data to display")  # Debug log

            # КАРТА - выводим в самом конце после всех текстовых элементов
            if maps_2gis_enabled and st.session_state.get("show_map", False):
                map_type = st.session_state.get("map_type", "points")
                
                if map_type == "points" and st.session_state.get("last_pydeck_data", []) and len(st.session_state["last_pydeck_data"]) > 0:
                    display_2gis_map(
                        pydeck_data=st.session_state["last_pydeck_data"],
                        map_type="points",
                        title="🗺️ Интерактивная карта 2GIS"
                    )
                
                elif map_type == "route" and st.session_state.get("path_points", []) and st.session_state.get("route_points", []):
                    display_2gis_map(
                        pydeck_data=[],  # Empty for route type
                        map_type="route",
                        path_points=st.session_state["path_points"],
                        route_points=st.session_state["route_points"],
                        title="🗺️ Построенный маршрут"
                    )

            # Сохраняем дополнительную информацию для истории сообщений
            st.session_state["messages"].append(
                {
                    "role": "assistant", 
                    "content": full_response_text,  # Теперь включает офферы в конце
                    "question": prompt,
                    "request_category": response.get("request_category", ""),
                    "show_map": st.session_state.get("show_map", False),
                    "map_type": st.session_state.get("map_type", "points"),
                    "pydeck_data": st.session_state.get("last_pydeck_data", []),
                    "places_text": places_text,
                    "aviasales_text": aviasales_text,
                    "offers_text": offers_text,
                    "table_data": table_data if 'table_data' in locals() else [],
                    "path_points": st.session_state.get("path_points", []),
                    "route_points": st.session_state.get("route_points", []),
                    "route_info": st.session_state.get("route_info", None),
                    "record_id": None,
                    "offers_link":offers_link if 'offers_link' in locals() else None
                }
            )
            
            # Добавляем оценку ответа
            col1, col2, col3 = st.columns(3)
            if col1.button("👍", key=f"thumbs_up_{len(st.session_state['messages'])}"):
                st.success("Вы поставили 👍")
            if col2.button("👎", key=f"thumbs_down_{len(st.session_state['messages'])}"):
                st.error("Вы поставили 👎")
            # Прямое создание кнопки-ссылки
            try:
                col3.link_button("🎁 Сгенерировать оффер", f"{offers_link}", key=f"generate_offer_{len(st.session_state['messages'])}")
            except:
                col3.markdown(f"[🎁 Сгенерировать оффер]({offers_link})")


            # Сохраняем в базу данных
            record_id = insert_chat_history_return_id(
                user_query=prompt,
                model_response=full_response_text,  # Сохраняем полный текст, включая информацию о местах
                mode=config["mode"],
                rating=None
            )

            # Обновляем record_id в сообщении
            st.session_state["messages"][-1]["record_id"] = record_id
