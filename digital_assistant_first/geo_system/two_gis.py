from digital_assistant_first.internet_search import *
import requests
import aiohttp
import json
import os


async def fetch_2gis_data(query, config):
    """
    Функция для получения данных из 2GIS API. 
    Извлекает информацию о местах/заведениях на основе запроса пользователя.
    """
    api_key = config.get("2gis-key", "")
    if not api_key:
        logger.warning("2GIS API ключ не указан в конфигурации.")
        return [], []
    
    # Извлекаем город из запроса
    city_extraction_prompt = """
    Определи название города из запроса пользователя. Верни ТОЛЬКО название города без каких-либо дополнительных слов или объяснений.
    Если город не указан явно, верни "Москва" как город по умолчанию.
    
    Запрос пользователя: {query}
    """
    
    messages = [
        {"role": "system", "content": city_extraction_prompt.format(query=query)}
    ]
    
    # Используем модель для определения города
    from langchain_openai import ChatOpenAI
    city_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    city_response = city_model.invoke(messages)
    city = city_response.content.strip()
    
    # Логируем определенный город для отладки
    logger.info(f"Извлечен город: {city} из запроса: {query}")
    
    try:
        # Формируем запрос к 2GIS API с явным указанием города
        base_url = "https://catalog.api.2gis.com/3.0/items"
        params = {
            "q": f"{query} {city}",  # Добавляем город к запросу
            "key": api_key,
            "fields": "items.point,items.full_address,items.name,items.reviews,items.contact_groups,items.address,items.address_name,items.address_comment,items.building_name,items.schedule,items.cuisine",
            "city": city  # Явно указываем город
        }
        
        # Логируем параметры запроса для отладки
        logger.info(f"Параметры запроса к 2GIS: {params}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("result", {}).get("items", [])
                    
                    if not results:
                        logger.warning(f"2GIS не вернул результатов для запроса: {query} в городе {city}")
                        return [], []
                    
                    # Логируем количество полученных результатов
                    logger.info(f"Получено {len(results)} результатов из 2GIS")
                    
                    # Добавляем детальное логирование структуры первого результата для отладки
                    if results:
                        logger.info(f"Структура первого результата: {json.dumps(results[0], ensure_ascii=False, indent=2)}")
                    
                    table_data = []
                    pydeck_data = []
                    
                    for item in results:
                        # Имя и адрес
                        name = item.get("name", "Без названия")
                        
                        # Улучшенное извлечение адреса - проверяем несколько полей
                        address = item.get("full_address", None)
                        if not address:
                            address = item.get("address_name", None)
                        if not address:
                            address = item.get("address", None)
                        if not address and "address_comment" in item:
                            address = item.get("address_comment")
                        if not address:
                            # Проверяем вложенные поля address
                            address_data = item.get("address", {})
                            if isinstance(address_data, dict):
                                address = address_data.get("text", "Адрес не указан")
                            
                        if not address:
                            address = "Адрес не указан"
                        
                        # Логируем данные адреса для каждого места
                        logger.info(f"Место: {name}, извлеченный адрес: {address}")
                        
                        # Геолокация
                        point = item.get("point", {})
                        lat = point.get("lat")
                        lon = point.get("lon")
                        
                        # Если нет координат, пропускаем
                        if not lat or not lon:
                            continue
                        
                        # Рейтинг и отзывы
                        reviews_data = item.get("reviews", {})
                        rating = reviews_data.get("general", {}).get("rating", 0)
                        reviews_count = reviews_data.get("general", {}).get("count", 0)
                        
                        # Извлечение типа кухни, если это ресторан
                        cuisine = "Не указано"
                        cuisine_data = item.get("cuisine", [])
                        if cuisine_data and len(cuisine_data) > 0:
                            cuisine_list = [c.get("name", "") for c in cuisine_data if c.get("name")]
                            cuisine = ", ".join(cuisine_list) if cuisine_list else "Не указано"
                        
                        # Извлечение графика работы
                        schedule_text = "Не указано"
                        schedule_data = item.get("schedule", {})
                        if schedule_data:
                            try:
                                if "work_time" in schedule_data:
                                    work_time = schedule_data.get("work_time", {})
                                    today_text = work_time.get("today", {}).get("text", "")
                                    schedule_text = today_text if today_text else "Не указано"
                                    
                                    # Если нет информации о сегодняшнем дне, проверяем общее расписание
                                    if not schedule_text or schedule_text == "Не указано":
                                        general_text = schedule_data.get("general", {}).get("text", "")
                                        schedule_text = general_text if general_text else "Не указано"
                            except Exception as e:
                                logger.error(f"Ошибка извлечения расписания: {str(e)}")
                                schedule_text = "Не указано"
                        
                        # Контактная информация
                        phone = ""
                        contact_groups = item.get("contact_groups", [])
                        for group in contact_groups:
                            if group.get("name") == "phone":
                                contacts = group.get("contacts", [])
                                if contacts:
                                    phone = contacts[0].get("value", "")
                        
                        # Добавляем данные для таблицы
                        table_entry = {
                            "name": name,
                            "address": address,
                            "rating": rating,
                            "reviews": reviews_count,
                            "phone": phone,
                            "lat": lat,
                            "lon": lon,
                            "cuisine": cuisine,
                            "schedule": schedule_text
                        }
                        table_data.append(table_entry)
                        
                        # Добавляем данные для карты
                        pydeck_entry = {
                            "name": name,
                            "lat": lat,
                            "lon": lon
                        }
                        pydeck_data.append(pydeck_entry)
                    
                    # Логируем данные о первом месте для отладки
                    if pydeck_data:
                        logger.info(f"Первая точка: {pydeck_data[0]['name']} - lat: {pydeck_data[0]['lat']}, lon: {pydeck_data[0]['lon']}")
                    
                    return table_data, pydeck_data
                else:
                    logger.error(f"Ошибка API 2GIS: {response.status}")
                    return [], []
    except Exception as e:
        logger.error(f"Ошибка при обработке данных 2GIS: {str(e)}")
        return [], []


async def build_route_2gis(start_point, end_point, config):
    """
    Функция для построения маршрута между двумя точками с использованием Routing API 2GIS.
    
    Args:
        start_point (dict): Словарь с координатами начальной точки {'lat': float, 'lon': float}
        end_point (dict): Словарь с координатами конечной точки {'lat': float, 'lon': float}
        config (dict): Конфигурация с API ключом
        
    Returns:
        tuple: (route_data, path_points, route_details) - данные о маршруте, точки для отображения и детали
    """
    api_key = config.get("2gis-key", "")
    if not api_key:
        logger.warning("2GIS API ключ не указан в конфигурации.")
        return None, [], {}
    
    try:
        # Проверяем расстояние между точками
        import math
        def calculate_distance(lat1, lon1, lat2, lon2):
            R = 6371  # радиус Земли в км
            dLat = math.radians(lat2 - lat1)
            dLon = math.radians(lon2 - lon1)
            a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            return distance
        
        distance = calculate_distance(start_point["lat"], start_point["lon"], end_point["lat"], end_point["lon"])
        logger.info(f"Расстояние между точками: {distance:.2f} км")
        
        # Если расстояние слишком большое (больше 100 км), создаем искусственный маршрут
        if distance > 100:
            logger.info(f"Расстояние превышает 100 км, создаем упрощенный маршрут между городами")
            # Создаем искусственную линию между точками с промежуточными точками
            num_points = max(2, min(10, int(distance / 50)))  # 1 точка на каждые 50 км, но не меньше 2 и не больше 10
            path_points = []
            
            for i in range(num_points):
                fraction = i / (num_points - 1)
                lat = start_point["lat"] + fraction * (end_point["lat"] - start_point["lat"])
                lon = start_point["lon"] + fraction * (end_point["lon"] - start_point["lon"])
                path_points.append({
                    "lon": lon, 
                    "lat": lat,
                    "color": "fast",  # Условно считаем межгородской маршрут быстрым
                    "style": "normal"
                })
            
            # Примерно рассчитываем данные о маршруте
            route_info = {
                "distance": int(distance * 1000),  # в метрах
                "duration": int(distance * 60 * 60 / 80),  # в секундах, при скорости 80 км/ч
                "has_traffic": False,
                "points_count": len(path_points)
            }
            
            # Создаем условные инструкции для межгородского маршрута
            steps = [
                {
                    "instruction": f"Направляйтесь из {start_point.get('name', 'начальной точки')} в {end_point.get('name', 'конечную точку')}",
                    "distance": int(distance * 1000),
                    "duration": int(distance * 60 * 60 / 80),
                    "street_name": "Межгородской маршрут"
                }
            ]
            
            route_details = {
                "steps": steps,
                "street_names": ["Межгородской маршрут"],
            }
            
            logger.info(f"Создан упрощенный маршрут длиной {route_info['distance']} м, {route_info['duration']} сек, {len(path_points)} точек")
            return route_info, path_points, route_details
        
        # Формируем запрос к Routing API 2GIS
        base_url = "https://routing.api.2gis.com/routing/7.0.0/global"
        
        # Собираем тело запроса
        request_body = {
            "points": [
                {
                    "type": "stop",
                    "lon": start_point.get("lon"),
                    "lat": start_point.get("lat")
                },
                {
                    "type": "stop",
                    "lon": end_point.get("lon"),
                    "lat": end_point.get("lat")
                }
            ],
            "locale": "ru",
            "transport": "driving",
            "route_mode": "fastest",
            "traffic_mode": "jam"
        }
        
        # Логируем параметры запроса для отладки
        logger.info(f"Запрос маршрута 2GIS: {json.dumps(request_body, ensure_ascii=False)}")
        
        async with aiohttp.ClientSession() as session:
            # Отправляем POST-запрос с телом
            async with session.post(
                f"{base_url}?key={api_key}", 
                json=request_body,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Добавляем подробное логирование ответа API
                    logger.info(f"Полный ответ API 2GIS Routing: {json.dumps(data, ensure_ascii=False, indent=2)}")
                    
                    # Сохраняем ответ API в JSON файл для анализа
                    json_file_path = os.path.join(os.path.dirname(__file__), "2gis_route_response.json")
                    try:
                        with open(json_file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        logger.info(f"Ответ API 2GIS сохранен в файл: {json_file_path}")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении ответа API в файл: {str(e)}")
                    
                    # Проверяем структуру ответа 
                    if not isinstance(data, dict):
                        logger.error(f"Неожиданный формат данных ответа: {type(data)}")
                        return None, [], {}
                    
                    # Анализируем структуру результата
                    result = data.get("result", None)
                    routes = []
                    
                    if isinstance(result, list):
                        # Если result - список, то это уже список маршрутов
                        routes = result
                        logger.info(f"API вернул список маршрутов напрямую, найдено {len(routes)} маршрутов")
                        
                        # Логируем структуру первого маршрута
                        if routes and len(routes) > 0:
                            logger.info(f"Ключи первого маршрута: {list(routes[0].keys())}")
                    elif isinstance(result, dict) and "routes" in result:
                        # Если result - словарь с ключом routes, то routes внутри
                        routes = result.get("routes", [])
                        logger.info(f"API вернул словарь с маршрутами, найдено {len(routes)} маршрутов")
                    else:
                        logger.error(f"Неожиданная структура ответа API: {data}")
                        return None, [], {}
                        
                    if not routes:
                        logger.warning("2GIS Routing API не вернул маршрутов")
                        return None, [], {}
                    
                    # Берем первый предложенный маршрут
                    route = routes[0]
                    
                    # Проверяем, является ли route словарем
                    if not isinstance(route, dict):
                        logger.error(f"Неожиданный формат данных маршрута: {type(route)}")
                        return None, [], {}
                    
                    # Извлекаем путевые точки и дополнительную информацию о сегментах
                    path_points = []
                    segments = []
                    street_names = []
                    
                    # Извлекаем названия улиц
                    if 'names' in route:
                        street_names = route.get('names', [])
                        logger.info(f"Извлечено {len(street_names)} названий улиц: {street_names}")
                    
                    # Создаем структуру для хранения инструкций по навигации
                    steps = []
                    
                    # Проверяем наличие маневров для инструкций
                    if 'maneuvers' in route:
                        maneuvers = route.get('maneuvers', [])
                        logger.info(f"Найдено {len(maneuvers)} маневров в маршруте")
                        
                        for i, maneuver in enumerate(maneuvers):
                            if isinstance(maneuver, dict):
                                instruction = maneuver.get('text', '')
                                distance = maneuver.get('distance', {}).get('value', 0) if isinstance(maneuver.get('distance'), dict) else maneuver.get('distance', 0)
                                duration = maneuver.get('duration', {}).get('value', 0) if isinstance(maneuver.get('duration'), dict) else maneuver.get('duration', 0)
                                
                                # Определяем название улицы для данного маневра
                                street_name = ""
                                if 'street_name' in maneuver:
                                    street_name = maneuver.get('street_name', '')
                                elif i < len(street_names):
                                    street_name = street_names[i]
                                
                                step = {
                                    "instruction": instruction,
                                    "distance": distance,
                                    "duration": duration,
                                    "street_name": street_name
                                }
                                steps.append(step)
                                
                                # Извлекаем точки пути для этого маневра
                                if 'outcoming_path' in maneuver:
                                    outcoming_path = maneuver.get('outcoming_path', {})
                                    if isinstance(outcoming_path, dict) and 'geometry' in outcoming_path:
                                        geometry_items = outcoming_path.get('geometry', [])
                                        
                                        for geo_item in geometry_items:
                                            if isinstance(geo_item, dict):
                                                # Извлекаем атрибуты сегмента
                                                color = geo_item.get('color', 'normal')  # fast, normal, slow
                                                style = geo_item.get('style', 'normal')  # normal, tunnel, bridge
                                                length = geo_item.get('length', 0)
                                                
                                                segment = {
                                                    "color": color,
                                                    "style": style,
                                                    "length": length
                                                }
                                                segments.append(segment)
                                                
                                                # Парсим точки из LINESTRING
                                                if 'selection' in geo_item:
                                                    selection = geo_item.get('selection', '')
                                                    if selection.startswith('LINESTRING('):
                                                        coord_str = selection[11:-1]  # Удаляем LINESTRING( и )
                                                        coords = coord_str.split(', ')
                                                        for coord in coords:
                                                            lon, lat = map(float, coord.split())
                                                            path_points.append({
                                                                "lon": lon,
                                                                "lat": lat,
                                                                "color": color,
                                                                "style": style
                                                            })
                    
                    # Если нет маневров, проверяем наличие геометрии маршрута напрямую
                    elif 'geometry' in route:
                        geometry = route.get('geometry', {})
                        if isinstance(geometry, dict) and 'coordinates' in geometry:
                            for path in geometry['coordinates']:
                                path_points.append({
                                    "lon": path[0],
                                    "lat": path[1],
                                    "color": "normal",
                                    "style": "normal"
                                })
                    
                    # Если не нашли ни маневров с геометрией, ни прямой геометрии,
                    # проверяем старый формат с maneuvers, но без маневров как шагов
                    elif 'maneuvers' in route:
                        maneuvers = route.get('maneuvers', [])
                        for maneuver in maneuvers:
                            if isinstance(maneuver, dict) and 'outcoming_path' in maneuver:
                                outcoming_path = maneuver.get('outcoming_path', {})
                                if isinstance(outcoming_path, dict) and 'geometry' in outcoming_path:
                                    geometry_items = outcoming_path.get('geometry', [])
                                    for geo_item in geometry_items:
                                        if isinstance(geo_item, dict):
                                            # Извлекаем атрибуты сегмента
                                            color = geo_item.get('color', 'normal')
                                            style = geo_item.get('style', 'normal')
                                            length = geo_item.get('length', 0)
                                            
                                            segment = {
                                                "color": color,
                                                "style": style,
                                                "length": length
                                            }
                                            segments.append(segment)
                                            
                                            # Парсим строку LINESTRING из selection
                                            if 'selection' in geo_item:
                                                selection = geo_item.get('selection', '')
                                                if selection.startswith('LINESTRING('):
                                                    # Извлекаем координаты из строки LINESTRING(...)
                                                    coord_str = selection[11:-1]  # Удаляем LINESTRING( и )
                                                    coords = coord_str.split(', ')
                                                    for coord in coords:
                                                        lon, lat = map(float, coord.split())
                                                        path_points.append({
                                                            "lon": lon,
                                                            "lat": lat,
                                                            "color": color,
                                                            "style": style
                                                        })
                    
                    # Логируем количество извлеченных точек маршрута
                    logger.info(f"Извлечено {len(path_points)} точек для маршрута")
                    logger.info(f"Извлечено {len(segments)} сегментов маршрута")
                    
                    if not path_points:
                        logger.warning("Не удалось извлечь точки маршрута из ответа API")
                        # Создаем искусственную прямую линию между точками
                        path_points = [
                            {"lon": start_point["lon"], "lat": start_point["lat"], "color": "normal", "style": "normal"},
                            {"lon": end_point["lon"], "lat": end_point["lat"], "color": "normal", "style": "normal"}
                        ]
                    
                    # Извлекаем информацию о маршруте
                    # Проверяем различные возможные поля для получения дистанции и времени
                    distance = 0
                    duration = 0
                    
                    if 'total_distance' in route:
                        total_distance = route.get('total_distance', {})
                        if isinstance(total_distance, dict) and 'value' in total_distance:
                            distance = total_distance.get('value', 0)
                        else:
                            distance = total_distance
                    elif 'distance' in route:
                        distance = route.get('distance', 0)
                            
                    if 'total_duration' in route:
                        total_duration = route.get('total_duration', {})
                        if isinstance(total_duration, dict) and 'value' in total_duration:
                            duration = total_duration.get('value', 0)
                        else:
                            duration = total_duration
                    elif 'duration' in route:
                        duration = route.get('duration', 0)
                    
                    # Если не смогли извлечь инструкции из маневров, создаем упрощенные
                    if not steps and segments:
                        current_distance = 0
                        current_segment_index = 0
                        
                        for segment in segments:
                            length = segment.get('length', 0)
                            current_distance += length
                            
                            # Каждые 500 метров или при смене типа сегмента создаем инструкцию
                            if current_distance >= 500 or current_segment_index + 1 == len(segments):
                                street_name = street_names[0] if street_names else "Неизвестная улица"
                                
                                step = {
                                    "instruction": f"Продолжайте движение по {street_name}",
                                    "distance": current_distance,
                                    "duration": int(current_distance / 10),  # приблизительно
                                    "street_name": street_name
                                }
                                steps.append(step)
                                current_distance = 0
                            
                            current_segment_index += 1
                    
                    route_info = {
                        "distance": distance,  # в метрах
                        "duration": duration,  # в секундах
                        "has_traffic": route.get("has_traffic", False),
                        "points_count": len(path_points)
                    }
                    
                    route_details = {
                        "steps": steps,
                        "street_names": street_names,
                        "segments": segments
                    }
                    
                    logger.info(f"Построен маршрут: {route_info['distance']} м, {route_info['duration']} сек, {len(path_points)} точек")
                    logger.info(f"Добавлено {len(steps)} шагов навигации")
                    
                    return route_info, path_points, route_details
                else:
                    # Логируем текст ошибки
                    error_text = await response.text()
                    logger.error(f"Ошибка Routing API 2GIS: статус {response.status}, ответ: {error_text}")
                    return None, [], {}
    except Exception as e:
        logger.error(f"Ошибка при запросе маршрута 2GIS: {str(e)}")
        # Добавляем стек вызовов для детального отслеживания ошибки
        import traceback
        logger.error(f"Стек вызовов: {traceback.format_exc()}")
        return None, [], {}


async def extract_route_points_from_query(query):
    """
    Извлекает начальную и конечную точки маршрута из запроса пользователя.
    
    Args:
        query (str): Запрос пользователя
        
    Returns:
        tuple: (start_query, end_query) - текстовые запросы для начальной и конечной точек
    """
    # Используем LLM для извлечения информации о маршруте
    from langchain_openai import ChatOpenAI
    
    extraction_prompt = """
    Извлеки начальную и конечную точки маршрута из запроса пользователя.
    Верни только два адреса или названия мест, разделенных символом "|", без дополнительных пояснений.
    Пример: "Красная площадь|Кремль"
    
    Запрос пользователя: {query}
    """
    
    messages = [
        {"role": "system", "content": extraction_prompt.format(query=query)}
    ]
    
    route_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    route_response = route_model.invoke(messages)
    
    # Разделяем ответ на начальную и конечную точки
    points_text = route_response.content.strip()
    points = points_text.split("|")
    
    if len(points) < 2:
        # Если не удалось разделить, используем эвристику
        # Ищем ключевые слова "из", "от", "в", "до", "между"
        if "из" in query.lower() and "в" in query.lower():
            parts = query.lower().split("из")[1].split("в")
            start_query = parts[0].strip()
            end_query = parts[1].strip()
        elif "от" in query.lower() and "до" in query.lower():
            parts = query.lower().split("от")[1].split("до")
            start_query = parts[0].strip()
            end_query = parts[1].strip()
        else:
            # Если не удалось найти ключевые слова, используем значения по умолчанию
            start_query = "Текущее местоположение"
            end_query = query
    else:
        start_query = points[0].strip()
        end_query = points[1].strip()
    
    logger.info(f"Извлечены точки маршрута: от '{start_query}' до '{end_query}'")
    
    return start_query, end_query


async def geocode_address(address, city, config):
    """
    Геокодирование адреса в координаты с помощью 2GIS API.
    
    Args:
        address (str): Адрес или название места
        city (str): Город для уточнения поиска
        config (dict): Конфигурация с API ключом
        
    Returns:
        dict: Координаты точки {'lat': float, 'lon': float, 'name': str}
    """
    api_key = config.get("2gis-key", "")
    if not api_key:
        logger.warning("2GIS API ключ не указан в конфигурации.")
        return None
    
    # Проверяем ключевые слова для точного определения локаций
    address_lower = address.lower()
    city_lower = city.lower()
    
    logger.info(f"Геокодирование: адрес='{address_lower}', город='{city_lower}'")
    
    # Жестко закодированные координаты для известных мест
    if "красная площадь" in address_lower and ("москва" in city_lower or "moscow" in city_lower):
        logger.info(f"Используем предопределенные координаты для Красной площади в Москве")
        return {
            "lat": 55.753930,
            "lon": 37.620795,
            "name": "Красная площадь",
            "address": "Москва, Красная площадь"
        }
    elif ("тверская улица" in address_lower or "тверская" in address_lower) and ("москва" in city_lower or "moscow" in city_lower):
        logger.info(f"Используем предопределенные координаты для Тверской улицы в Москве")
        return {
            "lat": 55.762188,
            "lon": 37.609619,
            "name": "Тверская улица",
            "address": "Москва, Тверская улица"
        }
    elif ("дворцовая площадь" in address_lower or "дворцовая" in address_lower) and ("санкт-петербург" in city_lower or "спб" in city_lower or "питер" in city_lower or "saint petersburg" in city_lower):
        logger.info(f"Используем предопределенные координаты для Дворцовой площади в Санкт-Петербурге")
        return {
            "lat": 59.939037,
            "lon": 30.315765,
            "name": "Дворцовая площадь",
            "address": "Санкт-Петербург, Дворцовая площадь"
        }
    
    try:
        # Используем search API для геокодирования
        base_url = "https://catalog.api.2gis.com/3.0/items"
        params = {
            "q": f"{address} {city}",
            "key": api_key,
            "fields": "items.point,items.name,items.full_address,items.city",
            "city": city
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("result", {}).get("items", [])
                    
                    if not results:
                        logger.warning(f"Не найдены координаты для адреса: {address} в городе {city}")
                        return None
                    
                    # Фильтруем результаты, чтобы они соответствовали запрошенному городу
                    if city.lower() in ['москва', 'moscow']:
                        # Ищем результаты с точным совпадением по городу
                        city_results = []
                        for item in results:
                            item_city = None
                            if 'city' in item and item['city']:
                                item_city = item['city']
                            elif 'full_address' in item and item['full_address']:
                                # Проверяем, содержит ли полный адрес название города
                                if 'москва' in item['full_address'].lower():
                                    item_city = 'Москва'
                            
                            if item_city and item_city.lower() in ['москва', 'moscow']:
                                city_results.append(item)
                        
                        if city_results:
                            logger.info(f"Найдено {len(city_results)} результатов в Москве для '{address}'")
                            results = city_results
                    
                    # То же самое для Санкт-Петербурга
                    elif city.lower() in ['санкт-петербург', 'питер', 'saint petersburg']:
                        city_results = []
                        for item in results:
                            item_city = None
                            if 'city' in item and item['city']:
                                item_city = item['city']
                            elif 'full_address' in item and item['full_address']:
                                if 'санкт-петербург' in item['full_address'].lower() or 'спб' in item['full_address'].lower():
                                    item_city = 'Санкт-Петербург'
                            
                            if item_city and item_city.lower() in ['санкт-петербург', 'спб', 'saint petersburg']:
                                city_results.append(item)
                        
                        if city_results:
                            logger.info(f"Найдено {len(city_results)} результатов в Санкт-Петербурге для '{address}'")
                            results = city_results
                    
                    # Берем первый найденный результат
                    item = results[0]
                    point = item.get("point", {})
                    
                    logger.info(f"Определены координаты для '{address}': lat={point.get('lat')}, lon={point.get('lon')}")
                    
                    return {
                        "lat": point.get("lat"),
                        "lon": point.get("lon"),
                        "name": item.get("name", "Точка"),
                        "address": item.get("full_address", "Адрес не указан")
                    }
                else:
                    logger.error(f"Ошибка геокодирования 2GIS: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Ошибка при геокодировании: {str(e)}")
        return None


async def build_route_from_query(query, config):
    """
    Построение маршрута на основе текстового запроса пользователя.
    
    Args:
        query (str): Запрос пользователя
        config (dict): Конфигурация с API ключом
        
    Returns:
        tuple: (route_info, path_points, points_data, route_details) - информация о маршруте, точки пути, 
               данные о начальной/конечной точках и детали маршрута (шаги, названия улиц и т.д.)
    """
    # Извлекаем город из запроса
    city_extraction_prompt = """
    Определи название города из запроса пользователя. Если в запросе упоминаются два города (например, "из Москвы в Санкт-Петербург"), верни оба города через запятую: "Москва, Санкт-Петербург".
    Верни ТОЛЬКО название города или города через запятую без каких-либо дополнительных слов или объяснений.
    Если город не указан явно, верни "Москва" как город по умолчанию.
    
    Запрос пользователя: {query}
    """
    
    messages = [
        {"role": "system", "content": city_extraction_prompt.format(query=query)}
    ]
    
    from langchain_openai import ChatOpenAI
    city_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    city_response = city_model.invoke(messages)
    cities = city_response.content.strip().split(',')
    
    # Очищаем города от пробелов
    cities = [city.strip() for city in cities]
    
    # Первый город для начальной точки, последний для конечной
    start_city = cities[0] if cities else "Москва"
    end_city = cities[-1] if len(cities) > 1 else start_city
    
    logger.info(f"Извлечены города: начальный='{start_city}', конечный='{end_city}'")
    
    # Извлекаем начальную и конечную точки маршрута
    start_query, end_query = await extract_route_points_from_query(query)
    
    # Преобразуем адреса в координаты
    start_point = await geocode_address(start_query, start_city, config)
    end_point = await geocode_address(end_query, end_city, config)
    
    if not start_point or not end_point:
        logger.error(f"Не удалось определить координаты точек маршрута: start={start_query} в {start_city}, end={end_query} в {end_city}")
        return None, [], [], {}
    
    # Строим маршрут между точками
    route_info, path_points, route_details = await build_route_2gis(start_point, end_point, config)
    
    if not route_info:
        logger.error(f"Не удалось построить маршрут между точками")
        return None, [], [], {}
    
    # Формируем данные о точках для отображения
    points_data = [
        {
            "name": f"Начало: {start_point['name']}",
            "lat": start_point["lat"],
            "lon": start_point["lon"],
            "is_start": True,
            "address": start_point.get("address", "Адрес не указан")
        },
        {
            "name": f"Конец: {end_point['name']}",
            "lat": end_point["lat"],
            "lon": end_point["lon"],
            "is_end": True,
            "address": end_point.get("address", "Адрес не указан")
        }
    ]
    
    # Формируем человекочитаемый текст с инструкциями
    instructions_text = []
    if route_details and "steps" in route_details and route_details["steps"]:
        for i, step in enumerate(route_details["steps"]):
            instruction = step.get("instruction", "")
            distance = step.get("distance", 0)
            street_name = step.get("street_name", "")
            
            # Форматируем расстояние
            if distance > 1000:
                distance_str = f"{distance/1000:.1f} км"
            else:
                distance_str = f"{distance} м"
            
            if instruction:
                step_text = f"{i+1}. {instruction}"
                if street_name and street_name not in instruction:
                    step_text += f" ({street_name})"
                if distance_str:
                    step_text += f" - {distance_str}"
                
                instructions_text.append(step_text)
    
    # Добавляем навигационные инструкции к информации о маршруте
    if instructions_text:
        route_details["instructions_text"] = instructions_text
    
    logger.info(f"Подготовлены данные маршрута: начало={points_data[0]['name']}, конец={points_data[1]['name']}")
    if instructions_text:
        logger.info(f"Сформировано {len(instructions_text)} навигационных инструкций")
    
    return route_info, path_points, points_data, route_details

