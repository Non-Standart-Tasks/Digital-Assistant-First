from digital_assistant_first.internet_search import *
import requests
import aiohttp


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
            "fields": "items.point,items.full_address,items.name,items.reviews,items.contact_groups",
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
                    
                    table_data = []
                    pydeck_data = []
                    
                    for item in results:
                        # Имя и адрес
                        name = item.get("name", "Без названия")
                        address = item.get("full_address", "Адрес не указан")
                        
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
                            "lon": lon
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

