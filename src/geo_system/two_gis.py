from src.internet_search import *
import requests


def fetch_2gis_data(user_input, config):
    """
    Получить данные из 2Гис Catalog API по заданному запросу.
    Возвращает два списка: для таблицы и для PyDeck-карты.
    """
    radius = 3000
    API_KEY = config['2gis-key']
    url = (
        "https://catalog.api.2gis.com/3.0/items"
        f"?q={user_input}"
        "&location=37.630866,55.752256"  # Центр Москвы
        f"&radius={radius}"
        "&fields=items.point,items.address,items.name,items.contact_groups,items.reviews,items.rating"
        f"&key={API_KEY}"
    )

    response = requests.get(url)
    data = response.json()
    items = data.get("result", {}).get("items", [])

    table_data = []
    pydeck_data = []
    for item in items:
        point = item.get("point")
        if point:
            lat = point.get("lat")
            lon = point.get("lon")
            # Для таблицы
            table_data.append({
                "Название": item.get("name", "Нет названия"),
                "Адрес": item.get("address_name", ""),
                "Рейтинг": item.get("reviews", {}).get("general_rating"),
                "Кол-во Отзывов": item.get("reviews", {}).get("org_review_count", 0),
            })
            # Для PyDeck
            pydeck_data.append({
                "name": item.get("name", "Нет названия"),
                "lat": lat,
                "lon": lon,
            })
    return table_data, pydeck_data

