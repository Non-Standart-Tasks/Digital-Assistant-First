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
from digital_assistant_first.internet_search import search_shopping, search_places
import pydeck as pdk

# Локальные импорты
from digital_assistant_first.telegram_system.telegram_rag import EnhancedRAGSystem
from digital_assistant_first.telegram_system.telegram_data_initializer import (
    TelegramManager,
)
from digital_assistant_first.telegram_system.telegram_initialization import (
    fetch_telegram_data,
)
from digital_assistant_first.utils.aviasales_parser import construct_aviasales_url
from digital_assistant_first.geo_system.two_gis import fetch_2gis_data
from digital_assistant_first.utils.yndx_restaurants import (
    analyze_restaurant_request,
    get_restaurants_by_category,
)
from digital_assistant_first.offergen.agent import validation_agent
from digital_assistant_first.offergen.utils import get_system_prompt_for_offers
from streamlit_app import initialize_model
from digital_assistant_first.yndx_system.restaurant_context import fetch_yndx_context
from digital_assistant_first.utils.link_ckecker import link_checker, corrector

from digital_assistant_first.utils.database import (
    init_db, 
    insert_chat_history_return_id, 
    update_chat_history_rating_by_id, 
    get_chat_record_by_id
)

logger = setup_logging(logging_path="logs/digital_assistant.log")
serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.csv")
init_db()

def aviasales_request(model, config, user_input):
    messages = [
        {"role": "system", "content": config["system_prompt_tickets"]},
        {"role": "user", "content": user_input},
    ]
    response = model.invoke(messages, stream=False)
    if hasattr(response, "content"):
        content = response.content
    elif hasattr(response, "message"):
        content = response.message.content
    else:
        content = str(response)
    analysis = content.strip()
    if analysis.startswith("```json"):
        analysis = analysis[7:]
    if analysis.endswith("```"):
        analysis = analysis[:-3]
    analysis = analysis.strip()
    tickets_need = json.loads(analysis)
    return tickets_need


def model_response_generator(model, config):
    """Сгенерировать ответ с использованием модели и ретривера."""
    user_input = st.session_state["messages"][-1]["content"]
    tickets_need = aviasales_request(model, config, user_input)
    #restaurant_context_text = fetch_yndx_context(user_input, model)
    try:
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

        # Если интернет-поиск включён, вызываем функции поиска, иначе возвращаем пустые строки
        if config.get("internet_search", False):
            _, serpapi_key = serpapi_key_manager.get_best_api_key()
            shopping_res = search_shopping(user_input, serpapi_key)
            internet_res, links, _ = search_places(user_input, serpapi_key)
        else:
            shopping_res = ""
            internet_res = ""
            links = ""

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

        if config.get("telegram_enabled", False):
            telegram_manager = TelegramManager()
            rag_system = EnhancedRAGSystem(
                data_file="data/telegram_messages.json", index_directory="data/"
            )
            telegram_context = fetch_telegram_data(user_input, rag_system, k=50)
        else:
            telegram_context = ""

        #if restaurant_context_text:
        #    restaurants_prompt = restaurant_context_text
        #else:
        #    restaurants_prompt = ""

        system_prompt_template = config["system_prompt"]
        formatted_prompt = system_prompt_template.format(
            context=message_history,
            internet_res=internet_res,
            links=links,
            shopping_res=shopping_res,
            telegram_context=telegram_context
        )
        # Если требуется получение данных по 2Гис, оставляем только table_data и pydeck_data
        table_data = []
        pydeck_data = []

        if config.get("mode") == "2Gis":
            table_data, pydeck_data = fetch_2gis_data(user_input, config)

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

        table_data = table_data if table_data else []
        pydeck_data = pydeck_data if pydeck_data else []

        # Проверка ссылок в ответе
        try:
            # Создаем event loop чтобы использовать асинхронные функции
            # TODO: переписать ВСЁ на asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            link_statuses = loop.run_until_complete(link_checker.run(answer))
            logger.info(f"Link statuses: {link_statuses}")

            if link_statuses.data.links:
                logger.info(
                    f"Original answer: {answer[:200]}..."
                )  # Log first 200 chars

                some_link_is_invalid = any(
                    not link.status for link in link_statuses.data.links
                )
                if some_link_is_invalid:
                    # Коррекция текста ответа
                    corrected_answer = loop.run_until_complete(
                        corrector.run(answer, deps=link_statuses.data.links)
                    )
                    answer = corrected_answer.data
                    logger.info(
                        f"Corrected answer: {answer[:200]}..."
                    )  # Log first 200 chars
        except Exception as e:
            logger.error(
                f"Error checking links: {str(e)}", exc_info=True
            )  # Include full traceback

        yield {
            "answer": answer,
            "aviasales_link": aviasales_url,
            "table_data": table_data,
            "pydeck_data": pydeck_data,
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


def handle_user_input(model, config, prompt):
    """Обработать пользовательский ввод и сгенерировать ответ ассистента."""
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
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

                if config["mode"] == "2Gis":

                    response_text += f"\n\n### Данные из 2Гис"
                    if "table_data" in chunk:
                        df = pd.DataFrame(chunk["table_data"])
                        st.dataframe(df)  # Красивое представление таблицы
                    else:
                        st.warning("Ничего не найдено.")

                    # Отрисовка PyDeck карты
                    if "pydeck_data" in chunk:
                        df_pydeck = pd.DataFrame(chunk["pydeck_data"])
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
                    else:
                        st.warning("Не найдено точек для отображения на PyDeck-карте.")

                # Update the response placeholder for each chunk, regardless of mode
                response_placeholder.markdown(response_text)

            st.session_state["messages"].append(
                {"role": "assistant", "content": response_text, "question": prompt}
            )

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
