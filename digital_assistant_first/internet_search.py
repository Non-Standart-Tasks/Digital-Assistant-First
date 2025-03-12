# Импорты стандартной библиотеки
import logging
import aiohttp
import json

from serpapi import GoogleSearch
from digital_assistant_first.utils.check_serp_response import APIKeyManager

# Локальные импорты
from digital_assistant_first.utils.paths import ROOT_DIR
from digital_assistant_first.utils.logging import setup_logging, log_api_call

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SerpApiClient:
    """Асинхронный клиент для работы с SerpAPI."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
        
    async def search(self, params):
        """Выполняет асинхронный запрос к SerpAPI."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"SerpAPI error: {response.status} - {error_text}")
                    raise Exception(f"SerpAPI request failed: {response.status}")

def search_map(q, coordinates, serpapi_key):
    try:
        # Проверяем, есть ли координаты и их значения
        if (
            not coordinates
            or not coordinates.get("latitude")
            or not coordinates.get("longitude")
        ):
            return []  # Возвращаем пустоту, если координаты отсутствуют

        latitude = coordinates.get("latitude")
        longitude = coordinates.get("longitude")
        zoom_level = "14z"  # Укажите необходимый уровень масштабирования карты

        # Формируем параметр ll из coordinates
        ll = f"@{latitude},{longitude},{zoom_level}"

        # Параметры запроса
        params = {"engine": "google_maps", "q": q, "ll": ll, "api_key": serpapi_key}

        search = GoogleSearch(params)
        results = search.get_dict()

        good_results = [
            [
                item.get("title", "Нет информации"),
                item.get("rating", "Нет информации"),
                item.get("reviews", "Нет информации"),
                item.get("address", "Нет информации"),
                item.get("website", "Нет информации"),
                item.get("phone", "Нет информации"),
            ]
            for item in results.get("local_results", [])
        ]

        log_api_call(
            logger=logger, source="SerpAPI Maps", request=q, response=good_results
        )

        return good_results

    except Exception as e:
        log_api_call(
            logger=logger, source="SerpAPI Maps", request=q, response="", error=str(e)
        )
        raise


async def search_shopping(q, serpapi_key):
    """Поиск товаров через serpapi."""
    try:
        params = {
            "engine": "google_shopping",
            "q": q,
            "api_key": serpapi_key,
            "hl": "ru",
            "gl": "ru",
            "tbs": "mr:1,price:1,ppr_max:15000",
        }
        client = SerpApiClient(api_key=serpapi_key)
        response = await client.search(params)

        # Извлекаем результаты поиска
        shopping_results = response.get("shopping_results", [])
        if not shopping_results:
            return ""

        # Форматируем результаты
        formatted_results = []
        for result in shopping_results[:5]:  # Берем только первые 5 результатов
            title = result.get("title", "No title")
            price = result.get("price", "No price")
            source = result.get("source", "No source")
            link = result.get("link", "")

            formatted_result = f"Название: {title}\nЦена: {price}\nМагазин: {source}\nСсылка: {link}\n"
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error in search_shopping: {str(e)}")
        return ""


async def search_places(q, serpapi_key):
    """Поиск мест через serpapi."""
    try:
        params = {
            "engine": "google",
            "q": q,
            "api_key": serpapi_key,
            "hl": "ru",
            "gl": "ru",
            "num": 8,
        }
        client = SerpApiClient(api_key=serpapi_key)
        response = await client.search(params)

        # Извлекаем результаты поиска
        organic_results = response.get("organic_results", [])
        if not organic_results:
            return "", "", {}

        # Форматируем результаты
        formatted_results = []
        links = []
        search_metadata = {}

        for i, result in enumerate(organic_results):
            title = result.get("title", f"Result {i+1}")
            link = result.get("link", "")
            snippet = result.get("snippet", "")

            if link:
                links.append(link)

            # Добавляем результат в форматированный список
            formatted_result = f"### {title}\n{snippet}\nСсылка: {link}\n"
            formatted_results.append(formatted_result)

        # Добавляем метаданные поиска
        if "search_metadata" in response:
            search_metadata = response["search_metadata"]

        return "\n".join(formatted_results), links, search_metadata
    except Exception as e:
        logger.error(f"Error in search_places: {str(e)}")
        return "", "", {}


async def yandex_search(q, serpapi_key):
    try:
        params = {
            "lr": "225",
            "engine": "yandex",
            "yandex_domain": "yandex.ru",
            "text": q,
            "api_key": serpapi_key,
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        results_with_titles_and_links = [
            (item["title"], item["link"], item["snippet"], item["displayed_link"])
            for item in results.get("organic_results", [])
            if "title" in item and "link" in item
        ]

        return results_with_titles_and_links

    except Exception as e:
        log_api_call(
            logger=logger, source="SerpAPI Yandex", request=q, response="", error=str(e)
        )
        raise
