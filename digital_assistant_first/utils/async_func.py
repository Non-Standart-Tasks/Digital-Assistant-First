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
        - маршруты (если запрос о том, как построить маршрут, проложить путь, найти дорогу между местами и т.п.)
        - магазины (если запрос о торговых центрах, магазинах, супермаркетах, скидках в них и т.п.)
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
    offers_data = {}  # Инициализируем как пустой словарь вместо пустого списка
    
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
    
    # Задача для построения маршрута (только для категории "маршруты")
    route_info = None
    path_points = []
    points_data = []
    if request_category == "маршруты":
        print(f"DEBUG async: Добавляю задачу построения маршрута для запроса: {user_input}")
        tasks.append(build_route_from_query(user_input, config))
    
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
        
        # Результаты построения маршрута
        if request_category == "маршруты":
            if result_index < len(results) and not isinstance(results[result_index], Exception):
                route_info, path_points, points_data, route_details = results[result_index]
                print(f"DEBUG async: Получены результаты маршрута: {route_info}, {len(path_points) if path_points else 0} точек")
                # Сохраняем данные о маршруте для отображения
                if route_info and path_points and len(path_points) > 0:
                    pydeck_data = points_data
                    # Устанавливаем флаги для отображения маршрута
                    st.session_state["route_info"] = route_info
                    st.session_state["path_points"] = path_points
                    st.session_state["route_points"] = points_data
                    st.session_state["route_details"] = route_details
                    st.session_state["map_type"] = "route"
                    st.session_state["show_map"] = True
                    print(f"DEBUG async: Сохранены данные маршрута в session_state")
                    
                    # Добавляем навигационные инструкции к ответу
                    if route_details and "instructions_text" in route_details and route_details["instructions_text"]:
                        instructions_text = "\n\n**Навигационные инструкции:**\n" + "\n".join(route_details["instructions_text"])
                        response_text += instructions_text
                else:
                    print(f"DEBUG async: Не удалось построить маршрут")
            elif result_index < len(results):
                print(f"DEBUG async: Ошибка построения маршрута: {results[result_index]}")
            else:
                print(f"DEBUG async: Задача построения маршрута не вернула результатов")
            result_index += 1
        
        # Если офферы включены в конфигурации, запускаем обработку предложений независимо от категории запроса
        if config.get("offers_enabled", False):
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
                offers_data = {}  # Инициализируем как пустой словарь вместо пустого списка
        
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
        
        # Проверяем, нужно ли использовать нативный веб-поиск OpenAI
        use_openai_web_search = config.get("use_openai_web_search", False)
        
        # Отладочный вывод
        print(f"DEBUG ASYNC - use_openai_web_search value: {use_openai_web_search}, type: {type(use_openai_web_search)}")
        print(f"DEBUG ASYNC - Full config keys: {list(config.keys())}")
        
        # Если это строка, конвертируем в булево значение
        if isinstance(use_openai_web_search, str):
            use_openai_web_search = use_openai_web_search.lower() == 'true'
            print(f"DEBUG ASYNC - After conversion: {use_openai_web_search}")
            
        web_search_context_size = config.get("web_search_context_size", "medium")
        
        # Получаем ответ от модели
        if use_openai_web_search:
            # Используем нативный веб-поиск OpenAI
            logger.info(f"Используем нативный веб-поиск OpenAI для запроса: {user_input}")
            
            # Преобразуем сообщения в формат OpenAI API
            openai_messages = []
            
            # Проверяем тип переменной messages
            print(f"DEBUG ASYNC - messages type: {type(messages)}")
            
            # Если messages это строка или другой простой тип, создаем сообщения напрямую
            if isinstance(messages, str) or not hasattr(messages, "__iter__"):
                openai_messages = [
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": f"User query: {user_input}\nAdditional context: "}
                ]
            else:
                # Если это итерируемый объект, проверяем каждый элемент
                for msg in messages:
                    if hasattr(msg, "role") and hasattr(msg, "content"):
                        # Объект LangChain с атрибутами role и content
                        openai_messages.append({"role": msg.role, "content": msg.content})
                    elif isinstance(msg, tuple) and len(msg) == 2:
                        # Кортеж (role, content)
                        openai_messages.append({"role": msg[0], "content": msg[1]})
                    elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                        # Уже в правильном формате
                        openai_messages.append(msg)
                    else:
                        # Если не удалось определить формат, создаем сообщения по умолчанию
                        openai_messages = [
                            {"role": "system", "content": formatted_prompt},
                            {"role": "user", "content": f"User query: {user_input}\nAdditional context: "}
                        ]
                        break
            
            # Отладочная информация о созданных сообщениях
            print(f"DEBUG ASYNC - Created openai_messages: {openai_messages}")
            
            # Получаем клиент OpenAI из конфигурации
            openai_client = config.get("openai_client", OpenAI())
            
            # Вызываем OpenAI API напрямую
            openai_response = openai_client.responses.create(
                model=config["Model"],
                tools=[{
                    "type": "web_search_preview",
                    "search_context_size": web_search_context_size
                }],
                input=user_input
            )
            
            # Эмулируем ответ LangChain для совместимости с остальным кодом
            class OpenAIResponseWrapper:
                def __init__(self, openai_response):
                    self.content = openai_response.output_text
                    
            response = OpenAIResponseWrapper(openai_response)
        else:
            # Используем стандартный подход без веб-поиска
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
            if config.get("offers_enabled", False):
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
            if response.get("request_category") in ["рестораны", "ивенты", "маршруты"]:
                print(f"DEBUG: Отрисовка карты: show_map={st.session_state.get('show_map')}, map_type={st.session_state.get('map_type')}")
                if st.session_state.get("show_map", False):
                    map_type = st.session_state.get("map_type", "points")
                    print(f"DEBUG: Тип карты: {map_type}")
                    
                    if map_type == "points" and st.session_state.get("last_pydeck_data", []) and len(st.session_state["last_pydeck_data"]) > 0:
                        # Отображение точек на карте (рестораны, ивенты)
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
                                                radiusMinPixels=6,  # Минимальный размер точки в пикселях (видна при отдалении)
                                                radiusMaxPixels=100,  # Максимальный размер при приближении
                                                radiusScale=0.8,  # Масштабный коэффициент
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
                    
                    elif map_type == "route" and st.session_state.get("path_points", []) and st.session_state.get("route_points", []):
                        # Отображение маршрута на карте
                        with st.container():
                            st.markdown("## ")
                            st.subheader("🗺️ Построенный маршрут")
                            st.markdown("---")
                            
                            # Подготовка данных для PathLayer
                            path_points = st.session_state["path_points"]
                            route_points = st.session_state["route_points"]
                            
                            # Создаем DataFrame для точек маршрута
                            df_route_points = pd.DataFrame(route_points)
                            
                            print(f"DEBUG: Данные маршрута: начало={route_points[0]['name']}, конец={route_points[1]['name']}")
                            
                            # Рассчитываем центр маршрута
                            center_lat = df_route_points["lat"].mean()
                            center_lon = df_route_points["lon"].mean()
                            
                            # Для зума посчитаем максимальное расстояние между точками
                            max_lat = df_route_points["lat"].max()
                            min_lat = df_route_points["lat"].min()
                            max_lon = df_route_points["lon"].max()
                            min_lon = df_route_points["lon"].min()
                            
                            # Определим зум на основе расстояния
                            lat_diff = max_lat - min_lat
                            lon_diff = max_lon - min_lon
                            zoom_level = 10
                            if lat_diff > 0.1 or lon_diff > 0.1:
                                zoom_level = 9
                            if lat_diff > 0.2 or lon_diff > 0.2:
                                zoom_level = 8
                            if lat_diff > 0.5 or lon_diff > 0.5:
                                zoom_level = 7
                            
                            print(f"DEBUG: Координаты центра: {center_lat}, {center_lon}, zoom={zoom_level}")
                            
                            # Создаем слои для карты
                            layers = []
                            
                            # Группируем точки по цвету и стилю для отображения разных сегментов
                            segments = {}
                            
                            # Группируем последовательные точки с одинаковыми атрибутами в единые сегменты
                            current_segment_key = None
                            current_segment_points = []
                            current_segment_info = {}
                            
                            # Обходим все точки и группируем их в сегменты по цвету и стилю
                            for point in path_points:
                                # Извлекаем цвет и стиль для сегмента
                                color = point.get("color", "normal")
                                style = point.get("style", "normal")
                                segment_key = f"{color}_{style}"
                                
                                # Собираем информацию для подсказки (tooltip)
                                street_name = point.get("street_name", "")
                                speed_type = {
                                    "fast": "Быстрый участок", 
                                    "normal": "Обычный участок", 
                                    "slow": "Медленный участок"
                                }.get(color, "Участок маршрута")
                                
                                # Если это начало нового сегмента или первая точка
                                if segment_key != current_segment_key:
                                    # Если уже есть накопленные точки, сохраняем предыдущий сегмент
                                    if current_segment_points:
                                        if current_segment_key not in segments:
                                            segments[current_segment_key] = []
                                        segments[current_segment_key].append({
                                            "path": current_segment_points,
                                            "name": current_segment_info.get("street_name", ""),
                                            "speed_type": current_segment_info.get("speed_type", ""),
                                            "style_type": current_segment_info.get("style_type", ""),
                                            "style_string": " (" + {"normal": "дорога", "tunnel": "туннель", "bridge": "мост"}.get(current_segment_info.get("style_type", "normal"), "дорога") + ")"
                                        })
                                    
                                    # Начинаем новый сегмент
                                    current_segment_key = segment_key
                                    current_segment_points = []
                                    current_segment_info = {
                                        "street_name": street_name,
                                        "speed_type": speed_type,
                                        "style_type": style
                                    }
                                elif street_name and not current_segment_info.get("street_name"):
                                    # Обновляем название улицы, если оно появилось
                                    current_segment_info["street_name"] = street_name
                                
                                # Добавляем точку в текущий сегмент
                                current_segment_points.append([point["lon"], point["lat"]])
                            
                            # Добавляем последний сегмент, если есть накопленные точки
                            if current_segment_points and current_segment_key:
                                if current_segment_key not in segments:
                                    segments[current_segment_key] = []
                                segments[current_segment_key].append({
                                    "path": current_segment_points,
                                    "name": current_segment_info.get("street_name", ""),
                                    "speed_type": current_segment_info.get("speed_type", ""),
                                    "style_type": current_segment_info.get("style_type", ""),
                                    "style_string": " (" + {"normal": "дорога", "tunnel": "туннель", "bridge": "мост"}.get(current_segment_info.get("style_type", "normal"), "дорога") + ")"
                                })
                            
                            # Если нет сегментов (маловероятно), создаем один общий
                            if not segments:
                                segment_key = "normal_normal"
                                segments[segment_key] = [{
                                    "path": [[p["lon"], p["lat"]] for p in path_points],
                                    "name": "Маршрут",
                                    "speed_type": "Обычный участок",
                                    "style_type": "normal",
                                    "style_string": " (дорога)"
                                }]
                            
                            # Создаем слой для каждого типа сегмента
                            for segment_key, paths in segments.items():
                                # Безопасное разделение ключа, обрабатываем случай с несколькими подчеркиваниями
                                parts = segment_key.split("_")
                                if len(parts) >= 2:
                                    color_type = parts[0]
                                    style_type = parts[-1]  # Берем последний элемент как стиль
                                else:
                                    # Если нет подчеркивания или только одна часть
                                    color_type = segment_key
                                    style_type = "normal"
                                
                                # Устанавливаем цвет в зависимости от типа сегмента
                                if color_type == "fast":
                                    segment_color = [0, 180, 0, 200]  # Зеленый для быстрых участков
                                elif color_type == "normal":
                                    segment_color = [255, 165, 0, 200]  # Оранжевый для обычных участков
                                elif color_type == "slow":
                                    segment_color = [255, 0, 0, 200]  # Красный для медленных участков
                                else:
                                    segment_color = [0, 0, 255, 200]  # Синий по умолчанию
                                
                                # Устанавливаем ширину и параметры линии в зависимости от стиля
                                width = 5
                                dash_array = None
                                
                                if style_type == "tunnel":
                                    width = 6
                                    dash_array = [2, 1]  # Пунктирная линия для тоннелей
                                elif style_type == "bridge":
                                    width = 6
                                
                                # Добавляем слой для сегмента
                                path_layer = pdk.Layer(
                                    "PathLayer",
                                    data=paths,
                                    get_path="path",
                                    get_width=width,
                                    get_color=segment_color,
                                    width_min_pixels=3,
                                    pickable=True,
                                    dash_array=dash_array
                                )
                                layers.append(path_layer)
                            
                            # Отображаем карту
                            st.pydeck_chart(
                                pdk.Deck(
                                    map_style=None,
                                    initial_view_state=pdk.ViewState(
                                        latitude=center_lat,
                                        longitude=center_lon,
                                        zoom=zoom_level,
                                    ),
                                    layers=layers,
                                    tooltip={
                                        "html": "<b>{name}</b><br/>{speed_type}{style_string}",
                                        "style": {
                                            "backgroundColor": "white", 
                                            "color": "black",
                                            "fontSize": "12px",
                                            "borderRadius": "4px",
                                            "padding": "5px"
                                        }
                                    },
                                )
                            )
                    else:
                        print(f"DEBUG: Условия для отображения карты не выполнены")
                else:
                    print(f"DEBUG: Флаг show_map не установлен")

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
