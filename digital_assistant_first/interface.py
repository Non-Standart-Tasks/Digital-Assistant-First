# digital_assistant_first/interface.py

import logging
import json
import asyncio
import streamlit as st
import pandas as pd

from digital_assistant_first.utils.logging import setup_logging, log_api_call
from digital_assistant_first.internet_search import search_shopping, search_places
from digital_assistant_first.yndx_system.restaurant_context import fetch_yndx_context
from digital_assistant_first.utils.link_checker import link_checker, corrector
from digital_assistant_first.utils.database import (
    init_db, 
    insert_chat_history_return_id, 
    update_chat_history_rating_by_id, 
    get_chat_record_by_id
)
from digital_assistant_first.geo_system.two_gis import fetch_2gis_data
from digital_assistant_first.offergen.agent import validation_agent
from digital_assistant_first.offergen.utils import get_system_prompt_for_offers
from digital_assistant_first.telegram_system.telegram_data_initializer import (
    TelegramManager,
)
from digital_assistant_first.telegram_system.telegram_rag import EnhancedRAGSystem
from digital_assistant_first.telegram_system.telegram_initialization import (
    fetch_telegram_data,
)
from digital_assistant_first.utils.aviasales_parser import AviasalesHandler
from langchain_core.prompts import ChatPromptTemplate

init_db()
logger = setup_logging(logging_path="logs/digital_assistant.log")


def model_response_generator(model, config):
    """
    Генерирует ответ с использованием модели и ретривера (поиск в интернете и т.д.).
    Возвращает генератор, yielding объект с ключами:
      - "answer": сам текст ответа
      - "aviasales_link": если нужно, ссылка на авиабилеты
      - "table_data": данные для таблички (в случае 2Гис)
      - "pydeck_data": данные для отрисовки на карте (в случае 2Гис)
    """
    user_input = st.session_state["messages"][-1]["content"]

    # Собираем контекст из Яндекс (рестораны)
    restaurant_context_text = fetch_yndx_context(user_input, model)
    try:
        # Формируем историю сообщений (без системных)
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

        # Если интернет-поиск включён
        if config.get("internet_search", False):
            # Пример: ваш serpapi_key_manager... 
            # Допустим, у нас есть serpapi_key = "xxx"
            serpapi_key = "YOUR_SERPAPI_KEY"
            shopping_res = search_shopping(user_input, serpapi_key)
            internet_res, links, _ = search_places(user_input, serpapi_key)
        else:
            shopping_res = ""
            internet_res = ""
            links = ""

        # Aviasales
        if config.get("aviasales_search", True):
            aviasales_tool = AviasalesHandler()
            tickets_need = aviasales_tool.aviasales_request(model, config, user_input)
            if tickets_need.get('response', '').lower() == 'true':
                aviasales_url = aviasales_tool.construct_aviasales_url(
                    from_city=tickets_need["departure_city"],
                    to_city=tickets_need["destination"],
                    depart_date=tickets_need["start_date"],
                    return_date=tickets_need["end_date"],
                    adult_passengers=tickets_need["adult_passengers"],
                    child_passengers=tickets_need["child_passengers"],
                    travel_class=tickets_need.get("travel_class", ""),
                )
                aviasales_flight_info = aviasales_tool.get_info_aviasales_url(aviasales_url=aviasales_url)
            else:
                aviasales_url = ""
                aviasales_flight_info = ""
        else:
            aviasales_url = ""
            aviasales_flight_info = ""

        # Telegram-контекст (RAG)
        if config.get("telegram_enabled", False):
            telegram_manager = TelegramManager()
            rag_system = EnhancedRAGSystem(
                data_file="data/telegram_messages.json", index_directory="data/"
            )
            telegram_context = fetch_telegram_data(user_input, rag_system, k=50)
        else:
            telegram_context = ""

        # Если есть контекст с ресторанами
        restaurants_prompt = restaurant_context_text if restaurant_context_text else ""

        # Системный промпт
        system_prompt_template = config["system_prompt"]
        formatted_prompt = system_prompt_template.format(
            context=message_history,
            internet_res=internet_res,
            links=links,
            shopping_res=shopping_res,
            telegram_context=telegram_context,
            yndx_restaurants=restaurants_prompt,
            aviasales_flight_info=aviasales_flight_info,
        )

        # Для режима 2Гис
        table_data = []
        pydeck_data = []
        if config.get("mode") == "2Gis":
            table_data, pydeck_data = fetch_2gis_data(user_input, config)

        # Формируем финальный промпт
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", formatted_prompt),
                ("human", "User query: {input}\nAdditional context: {context}"),
            ]
        )
        messages = prompt_template.format(input=user_input, context="")

        response = model.invoke(messages, stream=True)
        if hasattr(response, "content"):
            answer = response.content
        elif hasattr(response, "message"):
            answer = response.message.content
        else:
            answer = str(response)

        # Проверяем ссылки в ответе
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            link_statuses = loop.run_until_complete(link_checker.run(answer))
            logger.info(f"Link statuses: {link_statuses}")

            if link_statuses.data.links:
                # если есть битые ссылки, коррекция текста
                some_link_is_invalid = any(not link.status for link in link_statuses.data.links)
                if some_link_is_invalid:
                    corrected_answer = loop.run_until_complete(
                        corrector.run(answer, deps=link_statuses.data.links)
                    )
                    answer = corrected_answer.data
        except Exception as e:
            logger.error(f"Error checking links: {str(e)}", exc_info=True)

        yield {
            "answer": answer,
            "aviasales_link": aviasales_url,
            "table_data": table_data,
            "pydeck_data": pydeck_data,
        }

        log_api_call(logger=logger, source=f"LLM ({config['Model']})", request=user_input, response=answer)

    except Exception as e:
        log_api_call(logger=logger, source=f"LLM ({config['Model']})", request=user_input, response="", error=str(e))
        raise


def handle_user_input(model, config, prompt):
    """
    Обработать пользовательский ввод:
      - Создать сообщение от пользователя
      - Генеративно получить ответ (через model_response_generator)
      - Сформировать финальный ответ (объединив чанки)
      - Сохранить запись в БД (insert_chat_history_return_id)
      - Добавить сообщение ассистента с record_id в сессию
    """
    # Добавляем в session_state
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерируем ответ
    response_text = ""
    aviasales_link = ""
    table_data = []
    pydeck_data = []

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        # Проверяем режим (Offers/2Gis/Chat)
        if config["mode"] == "Offers":
            # Генерация офферов
            try:
                # Определяем через validation_agent, сколько офферов надо и т.д.
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                validation_result = loop.run_until_complete(validation_agent.run(prompt)).data

                if validation_result.number_of_offers_to_generate < 1:
                    validation_result.number_of_offers_to_generate = 10

                system_prompt = get_system_prompt_for_offers(validation_result, prompt)
                if system_prompt == "No relevant offers were found for the search request.":
                    st.warning("Офферы не найдены, попробуйте уточнить запрос.")
                    return

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                response = model.invoke(messages, stream=True)

                # Постепенный вывод офферов (chunk-wise)
                for chunk in response:
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        key, value = chunk
                        if key == "content":
                            response_text += value
                            response_placeholder.markdown(response_text)

            except Exception as e:
                st.error(f"Error in offers generation: {str(e)}")
                return

        else:
            # Обычный чат / Режим 2Гис
            for chunk in model_response_generator(model, config):
                # Приходит кусок chunk["answer"]
                response_text += chunk["answer"]
                if "aviasales_link" in chunk:
                    if chunk["aviasales_link"].strip():
                        aviasales_link = chunk["aviasales_link"]

                # В режиме 2Гис добавляем табличные данные и карту
                if config["mode"] == "2Gis":
                    if chunk["table_data"]:
                        table_data = chunk["table_data"]
                    if chunk["pydeck_data"]:
                        pydeck_data = chunk["pydeck_data"]

                response_placeholder.markdown(response_text)

    # Если есть ссылка на авиасейлс, добавим её к ответу
    if aviasales_link:
        response_text += f"\n\n### Данные из Авиасейлс\n**Ссылка**: {aviasales_link}"

    # Отдельно отобразим таблицу и карту для 2Гис (после полного ответа)
    if config["mode"] == "2Gis":
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df)
        else:
            st.warning("Ничего не найдено в 2Гис.")

        if pydeck_data:
            import pydeck as pdk
            df_map = pd.DataFrame(pydeck_data)
            st.subheader("Карта")
            st.pydeck_chart(
                pdk.Deck(
                    map_style=None,
                    initial_view_state=pdk.ViewState(
                        latitude=df_map["lat"].mean(),
                        longitude=df_map["lon"].mean(),
                        zoom=13,
                    ),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=df_map,
                            get_position="[lon, lat]",
                            get_radius=30,
                            get_fill_color=[255, 0, 0],
                            pickable=True,
                        )
                    ],
                    tooltip={"html": "<b>{name}</b>", "style": {"color": "white"}},
                )
            )
        else:
            st.warning("Нет точек для PyDeck.")

    # Сохраняем ответ ассистента в session_state, чтобы отобразить его в истории
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": response_text,
            "question": prompt,  # при желании
        }
    )

    # Запишем в БД новую запись (user_query -> response_text)
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
        # Первое "system" сообщение
        st.session_state["messages"].append({"role": "system", "content": template_prompt})


def display_chat_history():
    """Отобразить историю чата из состояния сессии (включая кнопки рейтинга для ассистента)."""
    for i, message in enumerate(st.session_state["messages"]):
        with st.chat_message(message["role"]):
            # Если было сохранено "question", покажем его как заголовок (опционально)
            if "question" in message:
                st.markdown(f"**Вопрос**: {message['question']}")

            # Основной контент сообщения
            st.markdown(message["content"])

            # Если это ассистент и есть record_id, рисуем кнопки рейтинга
            if message["role"] == "assistant":
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