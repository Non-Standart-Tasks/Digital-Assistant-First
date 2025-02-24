# Импорты стандартной библиотеки
import logging

from serpapi.google_search import GoogleSearch
from src.utils.check_serp_response import APIKeyManager

# Локальные импорты
from src.utils.paths import ROOT_DIR
from src.utils.logging import setup_logging, log_api_call

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

serpapi_key_manager = APIKeyManager(path_to_file=ROOT_DIR / "api_keys_status.xlsx")


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


def search_shopping(q, serpapi_key):
    try:
        params = {"engine": "google_shopping", "q": q, "api_key": serpapi_key}

        search = GoogleSearch(params)
        results = search.get_dict()

        results_with_titles_and_links = [
            (item["title"], item["link"])
            for item in results.get("organic_results", [])
            if "title" in item and "link" in item
        ]

        log_api_call(
            logger=logger,
            source="SerpAPI Shopping",
            request=q,
            response=results_with_titles_and_links,
        )

        return results_with_titles_and_links

    except Exception as e:
        log_api_call(
            logger=logger,
            source="SerpAPI Shopping",
            request=q,
            response="",
            error=str(e),
        )
        raise


def search_places(q, serpapi_key):
    """Search for places using Google Search API, возвращает только первые 5 результатов."""
    try:
        params = {
            "q": q,
            #'location': 'Russia',
            "hl": "ru",
            "gl": "ru",
            "google_domain": "google.com",
            "api_key": serpapi_key,
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        good_results = [
            item["snippet"]
            for item in results.get("organic_results", [])
            if "snippet" in item
        ]

        results_with_titles_and_links = [
            (item["title"], item["link"])
            for item in results.get("organic_results", [])
            if "title" in item and "link" in item
        ]

        coordinates = results.get("local_map", {}).get("gps_coordinates", None)

        log_api_call(
            logger=logger,
            source="SerpAPI Places",
            request=q,
            response=good_results + results_with_titles_and_links,
        )

        return good_results, results_with_titles_and_links, coordinates

    except Exception as e:
        log_api_call(
            logger=logger, source="SerpAPI Places", request=q, response="", error=str(e)
        )
        raise


def yandex_search(q, serpapi_key):
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
